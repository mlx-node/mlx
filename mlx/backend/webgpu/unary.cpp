// Copyright 2026 Apple Inc.
//
// WebGPU unary element-wise operations.
//
// Uses dynamic WGSL code generation since WGSL lacks templates.
// Each unique (op, type, variant) combination produces a distinct shader
// module and compute pipeline, cached for reuse.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/op_exprs.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// C++ struct matching the WGSL UnaryParams layout (vec4-aligned).
// Total: 96 bytes.
struct UnaryParams {
  uint32_t size_ndim[4]; // [size, ndim, pad, pad]
  uint32_t offsets[4]; // [in_offset, out_offset, pad, pad]
  uint32_t shape_0[4]; // shape[0..3]
  uint32_t shape_1[4]; // shape[4..7]
  int32_t strides_0[4]; // strides[0..3]
  int32_t strides_1[4]; // strides[4..7]
};

// ---------------------------------------------------------------------------
// WGSL code generation
// ---------------------------------------------------------------------------

// Generate a WGSL compute shader for a unary operation.
// Parameters:
//   entry_name - unique name for the @compute entry point
//   in_type    - WGSL type for input
//   out_type   - WGSL type for output
//   op_expr    - WGSL expression using in_val
//   variant    - "v" (contiguous) or "g" (general/strided)
std::string make_unary_kernel(
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

  s << "struct UnaryParams {\n"
    << "  size_ndim: vec4<u32>,\n"
    << "  offsets: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  strides_0: vec4<i32>,\n"
    << "  strides_1: vec4<i32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << out_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: UnaryParams;\n\n";

  // General variant needs elem_to_loc helper
  if (variant == "g") {
    s << "fn get_shape(i: u32) -> u32 {\n"
      << "  if (i < 4u) { return params.shape_0[i]; }\n"
      << "  return params.shape_1[i - 4u];\n"
      << "}\n\n"
      << "fn get_stride(i: u32) -> i32 {\n"
      << "  if (i < 4u) { return params.strides_0[i]; }\n"
      << "  return params.strides_1[i - 4u];\n"
      << "}\n\n"
      << "fn elem_to_loc(idx: u32, ndim: u32) -> i32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return loc;\n"
      << "}\n\n";
  }

  s << wgpu::wgsl_linear_thread_id();

  // Entry point
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let size = params.size_ndim.x;\n"
    << "  let base = linear_thread_id(wg_id, lid, nwg) * N_READS;\n"
    << "  if (base >= size) { return; }\n"
    << "  let end = min(base + N_READS, size);\n";

  s << "  let in_off = params.offsets.x;\n"
    << "  let out_off = params.offsets.y;\n";

  if (variant == "g") {
    s << "  let ndim = params.size_ndim.y;\n"
      << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    let in_idx = u32(elem_to_loc(i, ndim) + i32(in_off));\n"
      << "    let in_val = input[in_idx];\n"
      << "    output[i + out_off] = " << op_expr << ";\n"
      << "  }\n";
  } else {
    s << "  for (var i: u32 = base; i < end; i = i + 1u) {\n"
      << "    let in_val = input[i + in_off];\n"
      << "    output[i + out_off] = " << op_expr << ";\n"
      << "  }\n";
  }

  s << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Operation expression lookup
// ---------------------------------------------------------------------------

// Get a short name for the operation (used in pipeline key / entry point).
const char* get_unary_short_name(const char* name) {
  std::string n(name);
  if (n == "Abs")
    return "abs_";
  if (n == "Negative")
    return "neg";
  if (n == "Exp")
    return "exp_";
  if (n == "Expm1")
    return "expm1";
  if (n == "Log")
    return "log_";
  if (n == "Log1p")
    return "log1p";
  if (n == "Sigmoid")
    return "sigmoid";
  if (n == "Sign")
    return "sign_";
  if (n == "Sin")
    return "sin_";
  if (n == "Cos")
    return "cos_";
  if (n == "Tan")
    return "tan_";
  if (n == "ArcSin")
    return "asin_";
  if (n == "ArcCos")
    return "acos_";
  if (n == "ArcTan")
    return "atan_";
  if (n == "Sinh")
    return "sinh_";
  if (n == "Cosh")
    return "cosh_";
  if (n == "Tanh")
    return "tanh_";
  if (n == "ArcSinh")
    return "asinh_";
  if (n == "ArcCosh")
    return "acosh_";
  if (n == "ArcTanh")
    return "atanh_";
  if (n == "Rsqrt")
    return "rsqrt";
  if (n == "Sqrt")
    return "sqrt_";
  if (n == "Square")
    return "square";
  if (n == "Ceil")
    return "ceil_";
  if (n == "Floor")
    return "floor_";
  if (n == "Round")
    return "round_";
  if (n == "Erf")
    return "erf_";
  if (n == "ErfInv")
    return "erfinv";
  if (n == "LogicalNot")
    return "lnot";
  if (n == "BitwiseInvert")
    return "bnot";
  return "unknown";
}

// Unary op WGSL expression builder lives in op_exprs.h as
// wgpu::get_unary_op_expr(). We alias it here for readability at the call
// site below.

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

void unary_op_gpu_dispatch(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    const Stream& s) {
  assert(inputs.size() >= 1);
  const auto& in = inputs[0];

  set_unary_output_data(in, out);
  // WebGPU: prevent buffer aliasing between input and output bindings
  wgpu::ensure_no_alias(out, in);
  wgpu::ensure_wgpu_size(out);

  if (out.size() == 0) {
    return;
  }

  bool contiguous = in.flags().contiguous;
  const char* variant = contiguous ? "v" : "g";

  const char* in_type = wgpu::dtype_to_wgsl_safe(in.dtype());
  const char* out_wgsl_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  const char* short_name = get_unary_short_name(op_name);

  std::string op_expr =
      wgpu::get_unary_op_expr(op_name, in_type, out_wgsl_type);

  // Build entry point name and pipeline key
  std::string entry_name = std::string("unary_") + short_name + "_" + variant +
      "_" + in_type + "_" + out_wgsl_type;
  // Get shader module with lazy codegen, then pipeline
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(entry_name, [&]() {
    return make_unary_kernel(
        entry_name, in_type, out_wgsl_type, op_expr, variant);
  });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Fill uniform buffer
  UnaryParams params{};
  uint32_t elem_count;

  // Convert byte offsets to element offsets (offset() returns bytes)
  params.offsets[0] = static_cast<uint32_t>(in.offset() / in.itemsize());
  params.offsets[1] = static_cast<uint32_t>(out.offset() / out.itemsize());

  if (!contiguous) {
    // General: collapse contiguous dims for efficiency
    auto [shape_collapsed, strides_collapsed] = collapse_contiguous_dims(in);

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
        params.strides_0[i] = static_cast<int32_t>(strides_collapsed[i]);
      } else {
        params.shape_1[i - 4] = static_cast<uint32_t>(shape_collapsed[i]);
        params.strides_1[i - 4] = static_cast<int32_t>(strides_collapsed[i]);
      }
    }
  } else {
    elem_count = static_cast<uint32_t>(out.data_size());
    params.size_ndim[0] = elem_count;
  }

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(UnaryParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpu::wgpu_bind_size(in);
  uint64_t out_buf_size = wgpu::wgpu_bind_size(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(UnaryParams)}});

  uint32_t total_threads = (elem_count + wgpu::N_READS - 1) / wgpu::N_READS;
  uint32_t num_workgroups =
      (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  auto [wg_x, wg_y] = wgpu::get_2d_grid(num_workgroups, "[WebGPU unary]");
  encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler(
      [uniform_buf]() { wgpu::device().uniform_pool().release(uniform_buf); });
}

} // namespace

// ---------------------------------------------------------------------------
// Primitive eval_gpu definitions via macro
// ---------------------------------------------------------------------------

#define UNARY_GPU(Op)                                               \
  void Op::eval_gpu(const std::vector<array>& inputs, array& out) { \
    auto& s = out.primitive().stream();                             \
    unary_op_gpu_dispatch(inputs, out, name(), s);                  \
  }

UNARY_GPU(Abs)
UNARY_GPU(Negative)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Log)
UNARY_GPU(Log1p)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Cos)
UNARY_GPU(Tan)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcTan)
UNARY_GPU(Sinh)
UNARY_GPU(Cosh)
UNARY_GPU(Tanh)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcTanh)
UNARY_GPU(Sqrt)
UNARY_GPU(Square)
UNARY_GPU(Ceil)
UNARY_GPU(Floor)
UNARY_GPU(Round)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(LogicalNot)
UNARY_GPU(BitwiseInvert)

#undef UNARY_GPU

} // namespace mlx::core
