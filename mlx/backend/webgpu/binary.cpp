// Copyright 2026 Apple Inc.
//
// WebGPU binary element-wise operations.
//
// Uses dynamic WGSL code generation since WGSL lacks templates.
// Each unique (op, type, variant) combination produces a distinct shader
// module and compute pipeline, cached for reuse.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// C++ struct matching the WGSL BinaryParams layout (vec4-aligned).
// Total: 128 bytes.
struct BinaryParams {
  uint32_t size_ndim[4]; // [size, ndim, pad, pad]
  uint32_t offsets[4];   // [a_offset, b_offset, out_offset, pad]
  uint32_t shape_0[4];   // shape[0..3]
  uint32_t shape_1[4];   // shape[4..7]
  int32_t a_strides_0[4]; // a_strides[0..3]
  int32_t a_strides_1[4]; // a_strides[4..7]
  int32_t b_strides_0[4]; // b_strides[0..3]
  int32_t b_strides_1[4]; // b_strides[4..7]
};

// ---------------------------------------------------------------------------
// WGSL code generation
// ---------------------------------------------------------------------------

// Generate a WGSL compute shader for a binary operation.
// Parameters:
//   entry_name - unique name for the @compute entry point
//   in_type    - WGSL type for inputs (e.g. "f32", "i32", "u32")
//   out_type   - WGSL type for output
//   op_expr    - WGSL expression using a_val and b_val
//   variant    - "ss", "sv", "vs", "vv", or "g"
std::string make_binary_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& out_type,
    const std::string& op_expr,
    const std::string& variant) {
  std::ostringstream s;

  if (in_type == "f16" || out_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "const WORKGROUP_SIZE: u32 = 256u;\n";
  s << "const N_READS: u32 = " << wgpu::N_READS << "u;\n\n";

  // Params struct (same layout for all variants; only .x/.y used for simple)
  s << "struct BinaryParams {\n"
    << "  size_ndim: vec4<u32>,\n"
    << "  offsets: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  a_strides_0: vec4<i32>,\n"
    << "  a_strides_1: vec4<i32>,\n"
    << "  b_strides_0: vec4<i32>,\n"
    << "  b_strides_1: vec4<i32>,\n"
    << "}\n\n";

  // Bindings
  s << "@group(0) @binding(0) var<storage, read> a: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read> b: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(2) var<storage, read_write> out: array<" << out_type
    << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: BinaryParams;\n\n";

  // General variant needs elem_to_loc helpers
  if (variant == "g") {
    s << "fn get_shape(i: u32) -> u32 {\n"
      << "  if (i < 4u) { return params.shape_0[i]; }\n"
      << "  return params.shape_1[i - 4u];\n"
      << "}\n\n"
      << "fn get_a_stride(i: u32) -> i32 {\n"
      << "  if (i < 4u) { return params.a_strides_0[i]; }\n"
      << "  return params.a_strides_1[i - 4u];\n"
      << "}\n\n"
      << "fn get_b_stride(i: u32) -> i32 {\n"
      << "  if (i < 4u) { return params.b_strides_0[i]; }\n"
      << "  return params.b_strides_1[i - 4u];\n"
      << "}\n\n"
      << "fn elem_to_loc_a(idx: u32, ndim: u32) -> i32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_a_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return loc;\n"
      << "}\n\n"
      << "fn elem_to_loc_b(idx: u32, ndim: u32) -> i32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_b_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return loc;\n"
      << "}\n\n";
  }

  // Entry point
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let size = params.size_ndim.x;\n"
    << "  let base = gid.x * N_READS;\n"
    << "  if (base >= size) { return; }\n"
    << "  let end = min(base + N_READS, size);\n";

  s << "  let a_off = params.offsets.x;\n"
    << "  let b_off = params.offsets.y;\n"
    << "  let out_off = params.offsets.z;\n";

  if (variant == "g") {
    s << "  let ndim = params.size_ndim.y;\n"
      << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    let a_idx = u32(elem_to_loc_a(i, ndim) + i32(a_off));\n"
      << "    let b_idx = u32(elem_to_loc_b(i, ndim) + i32(b_off));\n"
      << "    let a_val = a[a_idx];\n"
      << "    let b_val = b[b_idx];\n"
      << "    out[i + out_off] = " << op_expr << ";\n"
      << "  }\n";
  } else if (variant == "ss") {
    // ss: both scalars, loop just writes same result N_READS times
    s << "  let a_val = a[a_off];\n"
      << "  let b_val = b[b_off];\n"
      << "  let result = " << op_expr << ";\n"
      << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    out[i + out_off] = result;\n"
      << "  }\n";
  } else if (variant == "sv") {
    s << "  let a_val = a[a_off];\n"
      << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    let b_val = b[i + b_off];\n"
      << "    out[i + out_off] = " << op_expr << ";\n"
      << "  }\n";
  } else if (variant == "vs") {
    s << "  let b_val = b[b_off];\n"
      << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    let a_val = a[i + a_off];\n"
      << "    out[i + out_off] = " << op_expr << ";\n"
      << "  }\n";
  } else { // vv
    s << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    let a_val = a[i + a_off];\n"
      << "    let b_val = b[i + b_off];\n"
      << "    out[i + out_off] = " << op_expr << ";\n"
      << "  }\n";
  }

  s << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Operation expression lookup
// ---------------------------------------------------------------------------

// Map BinaryOpType to variant string suffix.
const char* binary_op_type_to_variant(BinaryOpType bopt) {
  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      return "ss";
    case BinaryOpType::ScalarVector:
      return "sv";
    case BinaryOpType::VectorScalar:
      return "vs";
    case BinaryOpType::VectorVector:
      return "vv";
    case BinaryOpType::General:
      return "g";
  }
  return "vv";
}

// Get a short name for the operation (used in pipeline key / entry point).
const char* get_op_short_name(const char* name) {
  std::string n(name);
  if (n == "Add") return "add";
  if (n == "Subtract") return "sub";
  if (n == "Multiply") return "mul";
  if (n == "Divide") return "div";
  if (n == "Remainder") return "rem";
  if (n == "Power") return "pow_";
  if (n == "Equal") return "eq";
  if (n == "NotEqual") return "ne";
  if (n == "Greater") return "gt";
  if (n == "GreaterEqual") return "ge";
  if (n == "Less") return "lt";
  if (n == "LessEqual") return "le";
  if (n == "LogicalAnd") return "land";
  if (n == "LogicalOr") return "lor";
  if (n == "Maximum") return "max_";
  if (n == "Minimum") return "min_";
  if (n == "LogAddExp") return "logaddexp";
  if (n == "ArcTan2") return "atan2_";
  return "unknown";
}

// Get the WGSL expression for an operation.
// References a_val and b_val as the two loaded input values.
// The out_type is used for explicit casts where needed.
std::string get_op_expr(
    const char* name,
    const std::string& in_type,
    const std::string& out_type) {
  std::string n(name);

  // Arithmetic ops - input and output types match
  if (n == "Add") return "a_val + b_val";
  if (n == "Subtract") return "a_val - b_val";
  if (n == "Multiply") return "a_val * b_val";
  if (n == "Divide") return "a_val / b_val";

  // Remainder: for float types use a - b*floor(a/b), for int just use %
  if (n == "Remainder") {
    if (in_type == "f32" || in_type == "f16") {
      return "a_val - b_val * floor(a_val / b_val)";
    }
    return "a_val % b_val";
  }

  // Power: pow() only works on float types
  if (n == "Power") {
    if (in_type == "f32" || in_type == "f16") {
      return "pow(a_val, b_val)";
    }
    // For integer power, we need a loop-based approach
    // But MLX typically promotes to float for power, so this is a fallback
    return out_type + "(pow(f32(a_val), f32(b_val)))";
  }

  // Comparison ops - output is u32 (bool stored as u32)
  if (n == "Equal")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val == b_val)";
  if (n == "NotEqual")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val != b_val)";
  if (n == "Greater")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val > b_val)";
  if (n == "GreaterEqual")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val >= b_val)";
  if (n == "Less")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val < b_val)";
  if (n == "LessEqual")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val <= b_val)";

  // Logical ops - inputs and outputs are bool-as-u32
  if (n == "LogicalAnd") {
    // For bool (u32): true if both non-zero
    if (in_type == "u32") {
      return "select(" + out_type + "(0), " + out_type +
          "(1), (a_val != 0u) && (b_val != 0u))";
    }
    // For other types, compare != 0
    return "select(" + out_type + "(0), " + out_type +
        "(1), (a_val != " + in_type + "(0)) && (b_val != " + in_type + "(0)))";
  }
  if (n == "LogicalOr") {
    if (in_type == "u32") {
      return "select(" + out_type + "(0), " + out_type +
          "(1), (a_val != 0u) || (b_val != 0u))";
    }
    return "select(" + out_type + "(0), " + out_type +
        "(1), (a_val != " + in_type + "(0)) || (b_val != " + in_type + "(0)))";
  }

  // Math ops
  if (n == "Maximum") return "max(a_val, b_val)";
  if (n == "Minimum") return "min(a_val, b_val)";

  if (n == "LogAddExp") {
    // logaddexp(a, b) = max(a,b) + log(1 + exp(-|a-b|))
    // Always compute in f32 for precision, cast back to out_type.
    return out_type +
        "(max(f32(a_val), f32(b_val)) + log(1.0 + exp(-abs(f32(a_val) - f32(b_val)))))";
  }

  if (n == "ArcTan2") {
    // atan2 only works on float types
    return out_type + "(atan2(f32(a_val), f32(b_val)))";
  }

  throw std::runtime_error(
      std::string("[WebGPU binary] Unknown op: ") + name);
}

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

void binary_op_gpu_dispatch(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    const Stream& s) {
  assert(inputs.size() >= 2);
  const auto& a = inputs[0];
  const auto& b = inputs[1];

  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  if (out.size() == 0) {
    return;
  }

  const char* in_type = wgpu::dtype_to_wgsl_safe(a.dtype());
  const char* out_wgsl_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  const char* variant = binary_op_type_to_variant(bopt);
  const char* short_name = get_op_short_name(op_name);

  std::string op_expr = get_op_expr(op_name, in_type, out_wgsl_type);

  // Build entry point name and pipeline key
  std::string entry_name = std::string("binary_") + short_name + "_" +
      variant + "_" + in_type + "_" + out_wgsl_type;
  // Get shader module with lazy codegen, then pipeline
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_binary_kernel(
            entry_name, in_type, out_wgsl_type, op_expr, variant);
      });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  // Fill uniform buffer
  BinaryParams params{};
  uint32_t elem_count;

  // Convert byte offsets to element offsets (offset() returns bytes)
  params.offsets[0] = static_cast<uint32_t>(a.offset() / a.itemsize());
  params.offsets[1] = static_cast<uint32_t>(b.offset() / b.itemsize());
  params.offsets[2] = static_cast<uint32_t>(out.offset() / out.itemsize());

  if (bopt == BinaryOpType::General) {
    auto [shape_collapsed, strides_vec] =
        collapse_contiguous_dims(a, b, out);
    auto& a_strides = strides_vec[0];
    auto& b_strides = strides_vec[1];

    uint32_t total_size = 1;
    for (auto& dim : shape_collapsed) {
      total_size *= static_cast<uint32_t>(dim);
    }
    elem_count = total_size;
    params.size_ndim[0] = total_size;
    params.size_ndim[1] = static_cast<uint32_t>(shape_collapsed.size());

    uint32_t ndim = params.size_ndim[1];
    for (uint32_t i = 0; i < ndim && i < wgpu::MAX_NDIM; ++i) {
      if (i < 4) {
        params.shape_0[i] = static_cast<uint32_t>(shape_collapsed[i]);
        params.a_strides_0[i] = static_cast<int32_t>(a_strides[i]);
        params.b_strides_0[i] = static_cast<int32_t>(b_strides[i]);
      } else {
        params.shape_1[i - 4] = static_cast<uint32_t>(shape_collapsed[i]);
        params.a_strides_1[i - 4] = static_cast<int32_t>(a_strides[i]);
        params.b_strides_1[i - 4] = static_cast<int32_t>(b_strides[i]);
      }
    }
  } else {
    elem_count = static_cast<uint32_t>(out.data_size());
    params.size_ndim[0] = elem_count;
  }

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(BinaryParams));

  WGPUBuffer a_buf = wgpu::wgpu_buffer(a);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(b);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t a_buf_size = wgpuBufferGetSize(a_buf);
  uint64_t b_buf_size = wgpuBufferGetSize(b_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{a_buf, a_buf_size},
       {b_buf, b_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(BinaryParams)}});

  uint32_t total_threads = (elem_count + wgpu::N_READS - 1) / wgpu::N_READS;
  uint32_t num_workgroups =
      (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace

// ---------------------------------------------------------------------------
// Primitive eval_gpu definitions via macro
// ---------------------------------------------------------------------------

#define BINARY_GPU(Op)                                                  \
  void Op::eval_gpu(const std::vector<array>& inputs, array& out) {     \
    auto& s = out.primitive().stream();                                 \
    binary_op_gpu_dispatch(inputs, out, name(), s);                     \
  }

BINARY_GPU(Add)
BINARY_GPU(Subtract)
BINARY_GPU(Multiply)
BINARY_GPU(Divide)
BINARY_GPU(Remainder)
BINARY_GPU(Power)
BINARY_GPU(Equal)
BINARY_GPU(NotEqual)
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)
BINARY_GPU(LogAddExp)
BINARY_GPU(ArcTan2)

#undef BINARY_GPU

} // namespace mlx::core
