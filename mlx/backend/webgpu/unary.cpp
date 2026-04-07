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
  uint32_t offsets[4];   // [in_offset, out_offset, pad, pad]
  uint32_t shape_0[4];   // shape[0..3]
  uint32_t shape_1[4];   // shape[4..7]
  int32_t strides_0[4];  // strides[0..3]
  int32_t strides_1[4];  // strides[4..7]
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

  s << "const WORKGROUP_SIZE: u32 = 256u;\n\n";

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

  // Entry point
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x;\n"
    << "  let size = params.size_ndim.x;\n"
    << "  if (idx >= size) { return; }\n";

  s << "  let in_off = params.offsets.x;\n"
    << "  let out_off = params.offsets.y;\n";

  if (variant == "g") {
    s << "  let ndim = params.size_ndim.y;\n"
      << "  let in_idx = u32(elem_to_loc(idx, ndim) + i32(in_off));\n"
      << "  let in_val = input[in_idx];\n";
  } else {
    s << "  let in_val = input[idx + in_off];\n";
  }

  s << "  output[idx + out_off] = " << op_expr << ";\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Operation expression lookup
// ---------------------------------------------------------------------------

// Get a short name for the operation (used in pipeline key / entry point).
const char* get_unary_short_name(const char* name) {
  std::string n(name);
  if (n == "Abs") return "abs_";
  if (n == "Negative") return "neg";
  if (n == "Exp") return "exp_";
  if (n == "Expm1") return "expm1";
  if (n == "Log") return "log_";
  if (n == "Log1p") return "log1p";
  if (n == "Sigmoid") return "sigmoid";
  if (n == "Sign") return "sign_";
  if (n == "Sin") return "sin_";
  if (n == "Cos") return "cos_";
  if (n == "Tan") return "tan_";
  if (n == "ArcSin") return "asin_";
  if (n == "ArcCos") return "acos_";
  if (n == "ArcTan") return "atan_";
  if (n == "Sinh") return "sinh_";
  if (n == "Cosh") return "cosh_";
  if (n == "Tanh") return "tanh_";
  if (n == "ArcSinh") return "asinh_";
  if (n == "ArcCosh") return "acosh_";
  if (n == "ArcTanh") return "atanh_";
  if (n == "Sqrt") return "sqrt_";
  if (n == "Square") return "square";
  if (n == "Ceil") return "ceil_";
  if (n == "Floor") return "floor_";
  if (n == "Round") return "round_";
  if (n == "Erf") return "erf_";
  if (n == "ErfInv") return "erfinv";
  if (n == "LogicalNot") return "lnot";
  if (n == "BitwiseInvert") return "bnot";
  return "unknown";
}

// Helper: is the WGSL type a float type?
bool is_float_type(const std::string& t) {
  return t == "f32" || t == "f16";
}

// Get the WGSL expression for a unary operation.
// References in_val as the loaded input value.
std::string get_unary_op_expr(
    const char* name,
    const std::string& in_type,
    const std::string& out_type) {
  std::string n(name);
  bool is_float = is_float_type(in_type);

  // abs() works on i32 and float types in WGSL; for u32, it's identity
  if (n == "Abs") {
    if (in_type == "u32") {
      return "in_val";
    }
    return "abs(in_val)";
  }

  if (n == "Negative") {
    if (in_type == "u32") {
      // Unsigned negation: reinterpret as bitwise (MLX shouldn't call this)
      return "u32(-i32(in_val))";
    }
    return "-in_val";
  }

  if (n == "Exp") {
    if (is_float) return "exp(in_val)";
    return out_type + "(exp(f32(in_val)))";
  }

  if (n == "Expm1") {
    // expm1(x) = exp(x) - 1
    if (in_type == "f32") return "exp(in_val) - 1.0";
    if (in_type == "f16") return "exp(in_val) - 1.0h";
    return out_type + "(exp(f32(in_val)) - 1.0)";
  }

  if (n == "Log") {
    if (is_float) return "log(in_val)";
    return out_type + "(log(f32(in_val)))";
  }

  if (n == "Log1p") {
    // log1p(x) = log(1 + x)
    if (in_type == "f32") return "log(1.0 + in_val)";
    if (in_type == "f16") return "log(1.0h + in_val)";
    return out_type + "(log(1.0 + f32(in_val)))";
  }

  if (n == "Sigmoid") {
    // sigmoid(x) = 1 / (1 + exp(-x))
    if (in_type == "f32") return "1.0 / (1.0 + exp(-in_val))";
    if (in_type == "f16") return "1.0h / (1.0h + exp(-in_val))";
    return out_type + "(1.0 / (1.0 + exp(-f32(in_val))))";
  }

  if (n == "Sign") {
    if (in_type == "u32") {
      return "select(0u, 1u, in_val > 0u)";
    }
    if (in_type == "i32") {
      return "select(select(-1, 1, in_val > 0), 0, in_val == 0)";
    }
    return "sign(in_val)";
  }

  // Trig and hyperbolic functions - work natively on float types.
  // For non-float inputs, cast to f32, compute, cast back.
  if (n == "Sin") {
    if (is_float) return "sin(in_val)";
    return out_type + "(sin(f32(in_val)))";
  }
  if (n == "Cos") {
    if (is_float) return "cos(in_val)";
    return out_type + "(cos(f32(in_val)))";
  }
  if (n == "Tan") {
    if (is_float) return "tan(in_val)";
    return out_type + "(tan(f32(in_val)))";
  }
  if (n == "ArcSin") {
    if (is_float) return "asin(in_val)";
    return out_type + "(asin(f32(in_val)))";
  }
  if (n == "ArcCos") {
    if (is_float) return "acos(in_val)";
    return out_type + "(acos(f32(in_val)))";
  }
  if (n == "ArcTan") {
    if (is_float) return "atan(in_val)";
    return out_type + "(atan(f32(in_val)))";
  }
  if (n == "Sinh") {
    if (is_float) return "sinh(in_val)";
    return out_type + "(sinh(f32(in_val)))";
  }
  if (n == "Cosh") {
    if (is_float) return "cosh(in_val)";
    return out_type + "(cosh(f32(in_val)))";
  }
  if (n == "Tanh") {
    if (is_float) return "tanh(in_val)";
    return out_type + "(tanh(f32(in_val)))";
  }
  if (n == "ArcSinh") {
    if (is_float) return "asinh(in_val)";
    return out_type + "(asinh(f32(in_val)))";
  }
  if (n == "ArcCosh") {
    if (is_float) return "acosh(in_val)";
    return out_type + "(acosh(f32(in_val)))";
  }
  if (n == "ArcTanh") {
    if (is_float) return "atanh(in_val)";
    return out_type + "(atanh(f32(in_val)))";
  }

  if (n == "Sqrt") {
    if (!is_float) {
      return out_type + "(sqrt(f32(in_val)))";
    }
    return "sqrt(in_val)";
  }

  if (n == "Square") return "in_val * in_val";

  // ceil/floor/round: identity for integer types
  if (n == "Ceil") {
    if (!is_float) return "in_val";
    return "ceil(in_val)";
  }
  if (n == "Floor") {
    if (!is_float) return "in_val";
    return "floor(in_val)";
  }
  if (n == "Round") {
    if (!is_float) return "in_val";
    return "round(in_val)";
  }

  if (n == "Erf") {
    // Erf approximation using Abramowitz and Stegun formula 7.1.26
    // erf(x) ~ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x*x)
    // where t = 1/(1 + 0.3275911*|x|)
    // Always compute in f32 for precision, cast result to out_type.
    return out_type +
        "(sign(f32(in_val)) * (1.0 - "
        "(0.254829592 * (1.0/(1.0+0.3275911*abs(f32(in_val)))) + "
        "-0.284496736 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),2.0) + "
        "1.421413741 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),3.0) + "
        "-1.453152027 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),4.0) + "
        "1.061405429 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),5.0)"
        ") * exp(-f32(in_val)*f32(in_val))))";
  }

  if (n == "ErfInv") {
    // ErfInv approximation using Winitzki's formula:
    // erfinv(x) ~ sign(x) * sqrt(sqrt((c + ln(1-x^2)/2)^2 - ln(1-x^2)/a) - (c + ln(1-x^2)/2))
    // where a = 0.147, c = 2/(pi*a)
    // Always compute in f32 for precision.
    return out_type +
        "(sign(f32(in_val)) * sqrt(sqrt("
        "pow(2.0/(3.14159265*0.147) + log(1.0 - f32(in_val)*f32(in_val))*0.5, 2.0)"
        " - log(1.0 - f32(in_val)*f32(in_val))/0.147"
        ") - (2.0/(3.14159265*0.147) + log(1.0 - f32(in_val)*f32(in_val))*0.5)))";
  }

  if (n == "LogicalNot") {
    // bool is stored as u32: 0 or 1
    return "select(" + out_type + "(1), " + out_type + "(0), in_val != 0u)";
  }

  if (n == "BitwiseInvert") {
    return "~in_val";
  }

  throw std::runtime_error(
      std::string("[WebGPU unary] Unknown op: ") + name);
}

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

  if (out.size() == 0) {
    return;
  }

  bool contiguous = in.flags().contiguous;
  const char* variant = contiguous ? "v" : "g";

  const char* in_type = wgpu::dtype_to_wgsl(in.dtype());
  const char* out_wgsl_type = wgpu::dtype_to_wgsl(out.dtype());
  const char* short_name = get_unary_short_name(op_name);

  std::string op_expr = get_unary_op_expr(op_name, in_type, out_wgsl_type);

  // Build entry point name and pipeline key
  std::string entry_name = std::string("unary_") + short_name + "_" +
      variant + "_" + in_type + "_" + out_wgsl_type;
  // Get shader module with lazy codegen, then pipeline
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
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
    auto [shape_collapsed, strides_collapsed] =
        collapse_contiguous_dims(in);

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

  WGPUBuffer uniform_buf =
      wgpu::create_uniform_buffer(&params, sizeof(UnaryParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(UnaryParams)}});

  uint32_t num_workgroups =
      (elem_count + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
}

} // namespace

// ---------------------------------------------------------------------------
// Primitive eval_gpu definitions via macro
// ---------------------------------------------------------------------------

#define UNARY_GPU(Op)                                                   \
  void Op::eval_gpu(const std::vector<array>& inputs, array& out) {     \
    auto& s = out.primitive().stream();                                 \
    unary_op_gpu_dispatch(inputs, out, name(), s);                      \
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
