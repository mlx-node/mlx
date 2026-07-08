// Copyright 2026 Apple Inc.
//
// WebGPU implementation of the Compiled primitive.
//
// The Compiled primitive represents a fused chain of element-wise operations
// produced by MLX's compile pass (see mlx/compile.cpp). Each Compiled node
// carries:
//   - inputs_   : the real inputs (some may be scalar constants).
//   - outputs_  : the final output arrays.
//   - tape_     : an in-order list of intermediate arrays, each with a
//                 fusable primitive (unary, binary, ternary Select,
//                 Broadcast, or AsType).
//
// eval_gpu emits a single WGSL compute shader that walks the tape inline in
// one kernel launch, mirroring the Metal and CUDA compiled backends. The
// kernel has two variants:
//   - contiguous ("c") : all inputs share the output's row-contiguous shape;
//                         indexing is linear.
//   - general    ("g") : inputs need per-input broadcast strides; each input
//                         gets its own elem_to_loc computation.
//
// Pipeline cache key: kernel_lib_ + variant + "_ndim<n>" for general.
//
// Fallback: if codegen encounters an unknown primitive or an unsupported
// dtype combination the call throws a std::runtime_error that names the
// offending primitive and the Compiled kernel name. Users can then disable
// compilation (mx.disable_compile()) to sidestep the path.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/op_exprs.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core {

namespace {

// Maximum inputs supported in a single fused kernel. Matches
// max_compile_arrays in mlx/compile.cpp so we can always emit a large-enough
// offsets table.
constexpr uint32_t MAX_FUSED_INPUTS = 24;
// Maximum outputs supported in a single fused kernel. Real Compiled nodes
// rarely exceed 4 outputs; 8 gives headroom without bloating the uniform.
constexpr uint32_t MAX_FUSED_OUTPUTS = 8;

// C++ struct matching the WGSL CompiledParams layout (vec4-aligned, 240 B).
// Fields:
//   size_ndim : [total_size, ndim, num_stride_inputs, pad]
//   shape_0/1 : output shape, split into two vec4<u32>
//   offsets   : flat array of 32 u32 offsets (element-indexed, not bytes).
//               Layout: [in_offsets[0..MAX_FUSED_INPUTS-1],
//                        out_offsets[0..MAX_FUSED_OUTPUTS-1]]
// Total size = 16 + 16 + 16 + 128 = 176 bytes. Padded to 256 by WGPU.
struct CompiledParams {
  uint32_t size_ndim[4];
  uint32_t shape_0[4];
  uint32_t shape_1[4];
  uint32_t offsets[MAX_FUSED_INPUTS + MAX_FUSED_OUTPUTS];
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Replace the `a_val` / `b_val` / `c_val` / `in_val` placeholders in a WGSL
// expression with the tmp-variable names for the tape step's inputs.
//
// For unary: substitute "in_val" -> tmp_name(inputs[0])
// For binary: substitute "a_val" -> tmp_name(inputs[0]),
//             "b_val" -> tmp_name(inputs[1])
// For select (ternary): emit "select(false, true, cond != 0u)" directly.
std::string substitute_unary(const std::string& expr, const std::string& in) {
  return wgpu::replace_all(expr, "in_val", "(" + in + ")");
}

std::string substitute_binary(
    const std::string& expr,
    const std::string& a,
    const std::string& b) {
  std::string out = wgpu::replace_all(expr, "a_val", "(" + a + ")");
  out = wgpu::replace_all(out, "b_val", "(" + b + ")");
  return out;
}

// Emit the WGSL representation of a constant. For integers we use the value
// as-is with the WGSL type suffix. For floats we rely on print_constant's
// textual output plus a cast to the input's WGSL type. Bool constants are
// written as 0u / 1u.
std::string ensure_float_suffix(std::string s, const std::string& wgsl_type) {
  bool has_dot = s.find('.') != std::string::npos;
  bool has_exp =
      s.find('e') != std::string::npos || s.find('E') != std::string::npos;
  if (!has_dot && !has_exp) {
    s += ".0";
  }
  if (wgsl_type == "f16") {
    s += "h";
  }
  return s;
}

std::string format_constant_for_wgsl(
    const array& x,
    const std::string& wgsl_type) {
  std::ostringstream ss;
  switch (x.dtype().val()) {
    case Dtype::Val::bool_:
      return x.item<bool>() ? "1u" : "0u";
    case Dtype::Val::uint32:
      ss << x.item<uint32_t>() << "u";
      return ss.str();
    case Dtype::Val::int32:
      ss << x.item<int32_t>();
      return ss.str();
    case Dtype::Val::float32: {
      print_float_constant<float>(ss, x);
      return ensure_float_suffix(ss.str(), "f32");
    }
    case Dtype::Val::float16: {
      print_float_constant<float16_t>(ss, x);
      return ensure_float_suffix(ss.str(), wgsl_type);
    }
    case Dtype::Val::bfloat16: {
      print_float_constant<bfloat16_t>(ss, x);
      // bf16 is stored as f32 on GPU; emit as f32 literal.
      return ensure_float_suffix(ss.str(), "f32");
    }
    default:
      throw std::runtime_error(
          std::string(
              "[WebGPU compiled] unsupported constant dtype for "
              "inlining in fused kernel (dtype=") +
          std::to_string(static_cast<int>(x.dtype().val())) + ")");
  }
}

// Translate a primitive name + input tmp names to a WGSL expression.
std::string build_tape_expression(
    const Primitive& prim,
    const std::vector<std::string>& in_tmps,
    const std::string& in_type,
    const std::string& out_type,
    const std::string& kernel_lib) {
  const char* name = prim.name();
  std::string n(name);

  // Broadcast / AsType: both are a per-element cast (identity in the first
  // case). Emit `out_type(tmp_in)`.
  if (is_static_cast(prim)) {
    if (in_tmps.size() != 1) {
      throw std::runtime_error(
          std::string(
              "[WebGPU compiled] static cast primitive with != 1 "
              "inputs in kernel ") +
          kernel_lib);
    }
    return out_type + "(" + in_tmps[0] + ")";
  }

  // Select (ternary): inputs are (cond, true_val, false_val).
  if (n == "Select") {
    if (in_tmps.size() != 3) {
      throw std::runtime_error(
          "[WebGPU compiled] Select with != 3 inputs in kernel " + kernel_lib);
    }
    // cond is bool (u32 on GPU); WGSL select picks arg1 when arg3 is true.
    return "select(" + in_tmps[2] + ", " + in_tmps[1] + ", (" + in_tmps[0] +
        ") != 0u)";
  }

  // Binary ops: check by looking up the expression in op_exprs.h. The lookup
  // throws if unknown, so we catch and rethrow with more context on failure.
  if (in_tmps.size() == 2) {
    try {
      std::string expr = wgpu::get_binary_op_expr(name, in_type, out_type);
      return substitute_binary(expr, in_tmps[0], in_tmps[1]);
    } catch (const std::runtime_error&) {
      // Fall through to unary (some ops e.g. LogicalNot are unary-only).
    }
  }

  // Unary ops
  if (in_tmps.size() == 1) {
    try {
      std::string expr = wgpu::get_unary_op_expr(name, in_type, out_type);
      return substitute_unary(expr, in_tmps[0]);
    } catch (const std::runtime_error&) {
      // Fall through to the catch-all throw below.
    }
  }

  throw std::runtime_error(
      std::string("[WebGPU compiled] Unsupported primitive '") + name +
      "' in fused kernel " + kernel_lib);
}

// ---------------------------------------------------------------------------
// WGSL kernel generation
// ---------------------------------------------------------------------------

std::string build_compiled_kernel(
    const std::string& entry_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    bool contiguous,
    uint32_t ndim,
    const std::string& kernel_lib) {
  NodeNamer namer;
  std::ostringstream s;

  // Determine whether we need the f16 extension.
  auto needs_f16 = [&](Dtype d) {
    return d == float16 && std::string(wgpu::dtype_to_wgsl_safe(d)) == "f16";
  };
  bool any_f16 = false;
  for (auto& x : inputs)
    any_f16 = any_f16 || needs_f16(x.dtype());
  for (auto& x : outputs)
    any_f16 = any_f16 || needs_f16(x.dtype());
  for (auto& x : tape)
    any_f16 = any_f16 || needs_f16(x.dtype());

  if (any_f16) {
    s << "enable f16;\n\n";
  }

  s << "const WORKGROUP_SIZE: u32 = 256u;\n";
  s << "const N_READS: u32 = " << wgpu::N_READS << "u;\n";
  s << "const MAX_NDIM: u32 = " << wgpu::MAX_NDIM << "u;\n\n";

  // Uniform block matches CompiledParams.
  s << "struct CompiledParams {\n"
    << "  size_ndim: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  offsets: array<vec4<u32>, "
    << ((MAX_FUSED_INPUTS + MAX_FUSED_OUTPUTS) / 4) << ">,\n"
    << "}\n\n";

  // Bindings: one storage-read buffer per non-constant input, then one
  // storage read_write buffer per output, then the uniform, then (for
  // strided) the strides storage buffer.
  uint32_t binding = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i))
      continue;
    const auto& x = inputs[i];
    auto& xname = namer.get_name(x);
    const char* wgsl_type = wgpu::dtype_to_wgsl_safe(x.dtype());
    s << "@group(0) @binding(" << binding++ << ") var<storage, read> " << xname
      << ": array<" << wgsl_type << ">;\n";
  }
  for (size_t o = 0; o < outputs.size(); ++o) {
    const auto& x = outputs[o];
    auto& xname = namer.get_name(x);
    const char* wgsl_type = wgpu::dtype_to_wgsl_safe(x.dtype());
    s << "@group(0) @binding(" << binding++ << ") var<storage, read_write> "
      << xname << ": array<" << wgsl_type << ">;\n";
  }
  s << "@group(0) @binding(" << binding++
    << ") var<uniform> params: CompiledParams;\n";
  bool have_strides = !contiguous;
  if (have_strides) {
    s << "@group(0) @binding(" << binding++
      << ") var<storage, read> in_strides: array<i32>;\n";
  }
  s << "\n";

  // Helpers: offsets accessors and (for strided) elem_to_loc.
  s << "fn get_shape(i: u32) -> u32 {\n"
    << "  if (i < 4u) { return params.shape_0[i]; }\n"
    << "  return params.shape_1[i - 4u];\n"
    << "}\n\n";
  s << "fn get_offset(i: u32) -> u32 {\n"
    << "  return params.offsets[i / 4u][i % 4u];\n"
    << "}\n\n";

  if (have_strides) {
    s << "fn elem_to_loc(idx: u32, stride_base: u32, ndim: u32) -> i32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var d: u32 = ndim - 1u; d < ndim; d = d - 1u) {\n"
      << "    let dim_size = get_shape(d);\n"
      << "    let dim_stride = in_strides[stride_base + d];\n"
      << "    loc += i32(idx_rem % dim_size) * dim_stride;\n"
      << "    idx_rem = idx_rem / dim_size;\n"
      << "  }\n"
      << "  return loc;\n"
      << "}\n\n";
  }

  s << wgpu::wgsl_linear_thread_id();

  // Compute entry point.
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let size = params.size_ndim.x;\n"
    << "  let base = linear_thread_id(wg_id, lid, nwg) * N_READS;\n"
    << "  if (base >= size) { return; }\n"
    << "  let end = min(base + N_READS, size);\n";

  if (have_strides) {
    s << "  let ndim = params.size_ndim.y;\n";
  }

  // Loop over N_READS elements per thread.
  s << "  for (var i: u32 = base; i < end; i = i + 1u) {\n";

  // Read the inputs into tmp variables.
  uint32_t in_offset_slot = 0; // index into offsets[] array for inputs
  uint32_t stride_input_idx = 0; // index into strided input list
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    auto& xname = namer.get_name(x);
    const char* wgsl_type = wgpu::dtype_to_wgsl_safe(x.dtype());
    std::string tmp_name = "tmp_" + xname;

    if (is_constant(i)) {
      std::string lit = format_constant_for_wgsl(x, wgsl_type);
      s << "    let " << tmp_name << ": " << wgsl_type << " = " << wgsl_type
        << "(" << lit << ");\n";
      continue;
    }

    // Non-constant inputs consume an offsets[] slot. The offset is the
    // array's base (arr.offset() in elements).
    uint32_t off_slot = in_offset_slot++;

    if (is_scalar(x)) {
      s << "    let " << tmp_name << ": " << wgsl_type << " = " << xname
        << "[get_offset(" << off_slot << "u)];\n";
      continue;
    }

    if (contiguous) {
      s << "    let " << tmp_name << ": " << wgsl_type << " = " << xname
        << "[i + get_offset(" << off_slot << "u)];\n";
    } else {
      // Strided: elem_to_loc using the per-input stride base.
      uint32_t stride_base = stride_input_idx * wgpu::MAX_NDIM;
      stride_input_idx++;
      s << "    let " << tmp_name << ": " << wgsl_type << " = " << xname
        << "[u32(elem_to_loc(i, " << stride_base << "u, ndim)"
        << " + i32(get_offset(" << off_slot << "u)))];\n";
    }
  }

  // Walk the tape, emitting a tmp_X = <expr>; for each step.
  for (const auto& step : tape) {
    auto& step_name = namer.get_name(step);
    std::string tmp_name = "tmp_" + step_name;
    const char* step_wgsl_type = wgpu::dtype_to_wgsl_safe(step.dtype());

    // Collect input tmp names.
    std::vector<std::string> in_tmps;
    in_tmps.reserve(step.inputs().size());
    for (auto& inp : step.inputs()) {
      in_tmps.push_back("tmp_" + namer.get_name(inp));
    }

    // CEL-F009: previously this picked `step.inputs()[0].dtype()`. That
    // mis-dispatches Select (inputs[0] is a bool mask) and any op where
    // the first tape input is a bool/u32 comparison result — the lookup
    // table in op_exprs.h would pick an integer code path for what is
    // actually a float compute. We now prefer the first non-bool input's
    // dtype for the hint, falling back to the step's own output dtype.
    // MLX promotes mixed-type binaries before the compile pass, so this
    // single representative type is safe to feed to get_binary_op_expr.
    std::string in_type_hint = step_wgsl_type;
    for (const auto& inp : step.inputs()) {
      if (inp.dtype() == bool_)
        continue;
      in_type_hint = wgpu::dtype_to_wgsl_safe(inp.dtype());
      break;
    }

    std::string expr = build_tape_expression(
        step.primitive(), in_tmps, in_type_hint, step_wgsl_type, kernel_lib);

    s << "    let " << tmp_name << ": " << step_wgsl_type << " = " << expr
      << ";\n";
  }

  // Write outputs. Their offsets are stored after the input offsets.
  uint32_t out_offset_slot = MAX_FUSED_INPUTS;
  for (size_t o = 0; o < outputs.size(); ++o) {
    const auto& x = outputs[o];
    auto& xname = namer.get_name(x);
    s << "    " << xname << "[i + get_offset(" << (out_offset_slot + o)
      << "u)] = tmp_" << xname << ";\n";
  }

  s << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Compiled::eval_gpu
// ---------------------------------------------------------------------------

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();

  // Bound-check the uniform layout.
  if (inputs_.size() > MAX_FUSED_INPUTS) {
    throw std::runtime_error(
        std::string("[WebGPU compiled] Too many fused inputs (") +
        std::to_string(inputs_.size()) + " > " +
        std::to_string(MAX_FUSED_INPUTS) + ") in kernel " + kernel_lib_);
  }
  if (outputs_.size() > MAX_FUSED_OUTPUTS) {
    throw std::runtime_error(
        std::string("[WebGPU compiled] Too many fused outputs (") +
        std::to_string(outputs_.size()) + " > " +
        std::to_string(MAX_FUSED_OUTPUTS) + ") in kernel " + kernel_lib_);
  }

  // Collapse contiguous dims to route to a faster kernel if possible.
  //
  // Upstream #3720: `contiguous` is now false whenever any input has
  // negative strides (e.g. from flip / negative-step slicing), which routes
  // those inputs to the strided ("g") kernel below. Metal/CUDA additionally
  // force 64-bit "large index" mode on `negative_strides` because their
  // small-index kernels did unsigned index math; WGSL has no i64, so large
  // mode cannot exist here. It is also not needed: the strided path is
  // already sign-correct — strides are bound as `array<i32>`, elem_to_loc
  // accumulates `var loc: i32` via `loc += i32(idx_rem % dim_size) *
  // dim_stride`, and the read site computes `u32(elem_to_loc(...) +
  // i32(get_offset(...)))`, where the sum is >= 0 for any in-bounds element.
  // The pre-existing i32-range limit on offsets applies to positive and
  // negative strides alike, so `negative_strides` needs no extra routing.
  auto [contiguous, negative_strides, shape, strides] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);
  (void)negative_strides;

  // Allocate outputs with input donation (same semantics as Metal / CPU).
  // Note: compiled_allocate_outputs uses the CPU itemsize. For dtypes where
  // the GPU element is wider (bool, bf16) we call ensure_wgpu_size below.
  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);
  for (auto& o : outputs) {
    wgpu::ensure_wgpu_size(o);
  }

  // WebGPU forbids the same buffer appearing as both a read-only and a
  // read-write binding in the same compute pass. When an output gets
  // donated from an input, break the alias.
  for (auto& o : outputs) {
    for (auto& in : inputs) {
      wgpu::ensure_no_alias(o, in);
    }
  }

  if (outputs[0].size() == 0) {
    return;
  }

  uint32_t ndim = contiguous ? 0u : static_cast<uint32_t>(shape.size());
  if (!contiguous && ndim > wgpu::MAX_NDIM) {
    throw std::runtime_error(
        std::string("[WebGPU compiled] ndim=") + std::to_string(ndim) +
        " exceeds MAX_NDIM=" + std::to_string(wgpu::MAX_NDIM) + " in kernel " +
        kernel_lib_);
  }

  // Pipeline cache key — encodes variant + ndim. Same tape with different
  // contiguity or different dim-count gets a distinct pipeline.
  //
  // CEL-F005: kernel_lib_ upstream is hashed over `kindof(dtype) +
  // itemsize(dtype)` per input/output. That collapses bfloat16 and float16
  // (both float-kind, both 2 bytes on CPU) even though they map to
  // *different* WGSL types on WebGPU (bf16 widens to f32 with a
  // value-preserving upload; f16 becomes `f16` when shader-f16 is enabled).
  // Without a distinguishing suffix, a Compiled node that first ran with
  // bf16 inputs would reuse its pipeline on the next f16 invocation, and
  // vice versa — either the WGSL validator rejects it or (worse) the
  // kernel silently runs with the wrong type. We append a per-input and
  // per-output dtype_val fingerprint so every dtype combination gets its
  // own pipeline key. This does NOT touch kernel_lib_ upstream; the
  // fingerprint is local to the WebGPU-side entry_name used for caching.
  std::string entry_name = kernel_lib_;
  entry_name += "_dt";
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (is_constant_(i))
      continue;
    entry_name +=
        "_i" + std::to_string(static_cast<int>(inputs_[i].dtype().val()));
  }
  for (const auto& t : tape_) {
    entry_name += "_t" + std::to_string(static_cast<int>(t.dtype().val()));
  }
  for (const auto& o : outputs_) {
    entry_name += "_o" + std::to_string(static_cast<int>(o.dtype().val()));
  }
  if (contiguous) {
    entry_name += "_c";
  } else {
    entry_name += "_g_nd" + std::to_string(ndim);
  }

  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(entry_name, [&]() {
    return build_compiled_kernel(
        entry_name,
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        contiguous,
        ndim,
        kernel_lib_);
  });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  // Populate uniform buffer.
  CompiledParams params{};
  uint32_t total_size;
  if (contiguous) {
    total_size = static_cast<uint32_t>(outputs[0].data_size());
  } else {
    total_size = 1;
    for (auto d : shape) {
      total_size *= static_cast<uint32_t>(d);
    }
    // Fill shape vec4s.
    for (uint32_t i = 0; i < ndim && i < wgpu::MAX_NDIM; ++i) {
      if (i < 4) {
        params.shape_0[i] = static_cast<uint32_t>(shape[i]);
      } else {
        params.shape_1[i - 4] = static_cast<uint32_t>(shape[i]);
      }
    }
  }
  params.size_ndim[0] = total_size;
  params.size_ndim[1] = ndim;

  // Populate per-input offsets.
  uint32_t in_slot = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i))
      continue;
    const auto& x = inputs[i];
    params.offsets[in_slot++] =
        static_cast<uint32_t>(x.offset() / x.itemsize());
  }
  uint32_t num_binding_inputs = in_slot;
  params.size_ndim[2] = num_binding_inputs;

  // Per-output offsets go after MAX_FUSED_INPUTS slots.
  for (size_t o = 0; o < outputs.size(); ++o) {
    params.offsets[MAX_FUSED_INPUTS + o] =
        static_cast<uint32_t>(outputs[o].offset() / outputs[o].itemsize());
  }

  // Strides storage buffer: one MAX_NDIM-sized block per strided input,
  // in the order matching strides_vec[1..] (which is how elem_to_loc looks
  // them up in the shader). strides_vec[0] is the output strides — skip it.
  WGPUBuffer strides_buf = nullptr;
  std::vector<int32_t> strides_data; // flat i32 array
  if (!contiguous) {
    uint32_t stride_input_count = 0;
    // strides contains (output_strides, in0_strides, in1_strides, ...)
    // matching the non-constant non-scalar inputs of the Compiled node.
    // We need to emit strides only for inputs that the shader treats as
    // strided (non-constant, non-scalar). Walk inputs_ in order, tracking
    // which ones were collapsed with non-trivial strides.
    // strides_vec from compiled_collapse_contiguous_dims follows that same
    // ordering (skipping constants and scalars), so we can iterate it in
    // lock-step.
    size_t strides_idx = 1; // [0] is output
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant_(i))
        continue;
      const auto& x = inputs[i];
      if (is_scalar(x))
        continue;
      // Pull the next strides row.
      if (strides_idx >= strides.size()) {
        throw std::runtime_error(
            std::string("[WebGPU compiled] strides count mismatch in kernel ") +
            kernel_lib_);
      }
      const auto& row = strides[strides_idx++];
      // Pad to MAX_NDIM with zeros.
      for (uint32_t d = 0; d < wgpu::MAX_NDIM; ++d) {
        int64_t sv = d < row.size() ? row[d] : 0;
        strides_data.push_back(static_cast<int32_t>(sv));
      }
      stride_input_count++;
    }
    if (stride_input_count == 0) {
      // No strided inputs — shouldn't happen for a non-contiguous path, but
      // be defensive: pad with a single row of zeros so the storage buffer
      // isn't empty (WebGPU storage buffers must have non-zero size).
      strides_data.assign(wgpu::MAX_NDIM, 0);
    }
    strides_buf = wgpu::create_storage_buffer(
        strides_data.data(), strides_data.size() * sizeof(int32_t));
  }

  auto& pool = dev.uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(dev.gpu_queue(), &params, sizeof(CompiledParams));

  // Build bind group: inputs, outputs, uniform, strides (if any).
  auto& encoder = wgpu::get_command_encoder(s);
  std::vector<std::pair<WGPUBuffer, uint64_t>> bg_entries;
  bg_entries.reserve(num_binding_inputs + outputs.size() + 2);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i))
      continue;
    const auto& x = inputs[i];
    encoder.set_input_array(x);
    WGPUBuffer buf = wgpu::wgpu_buffer(x);
    bg_entries.emplace_back(buf, wgpu::wgpu_bind_size(x));
  }
  for (auto& o : outputs) {
    encoder.set_output_array(o);
    WGPUBuffer buf = wgpu::wgpu_buffer(o);
    bg_entries.emplace_back(buf, wgpu::wgpu_bind_size(o));
  }
  bg_entries.emplace_back(uniform_buf, sizeof(CompiledParams));
  if (strides_buf) {
    bg_entries.emplace_back(strides_buf, wgpuBufferGetSize(strides_buf));
  }

  WGPUBindGroup bg;
  try {
    bg = wgpu::create_bind_group(pe.layout, bg_entries);
  } catch (const std::exception& e) {
    // Tag the opaque "Failed to create bind group" with the kernel name,
    // binding count, and input/output split so we can diagnose fused-node
    // layout mismatches without rebuilding with a debug shader. Also dumps
    // the first few binding buffer handles so we can see if the same
    // underlying WGPUBuffer is being bound to multiple slots (which some
    // WebGPU validators reject for storage bindings).
    std::ostringstream diag;
    diag << "[WebGPU compiled] " << e.what() << " kernel=" << kernel_lib_
         << " (" << (contiguous ? "c" : "g_nd")
         << (contiguous ? "" : std::to_string(ndim))
         << ") entries=" << bg_entries.size()
         << " inputs=" << num_binding_inputs << " outputs=" << outputs.size()
         << " has_strides=" << (strides_buf ? 1 : 0);
    diag << " bufs=[";
    for (size_t i = 0; i < bg_entries.size() && i < 12; ++i) {
      if (i)
        diag << ",";
      diag << reinterpret_cast<uintptr_t>(bg_entries[i].first);
    }
    diag << "]";
    throw std::runtime_error(diag.str());
  }

  uint32_t total_threads = (total_size + wgpu::N_READS - 1) / wgpu::N_READS;
  uint32_t num_workgroups =
      (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  auto [wg_x, wg_y] = wgpu::get_2d_grid(num_workgroups, "[WebGPU compiled]");
  encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf, strides_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
    if (strides_buf) {
      wgpuBufferDestroy(strides_buf);
      wgpuBufferRelease(strides_buf);
    }
  });
}

} // namespace mlx::core
