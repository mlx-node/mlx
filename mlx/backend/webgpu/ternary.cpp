// Copyright 2026 Apple Inc.
//
// WebGPU ternary (Select/Where) operation.
//
// out = condition ? true_val : false_val
//
// Uses dynamic WGSL code generation. Each unique (type, variant) combination
// produces a distinct shader module and compute pipeline, cached for reuse.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// C++ struct matching the WGSL TernaryParams layout (vec4-aligned).
// Total: 160 bytes.
struct TernaryParams {
  uint32_t size_ndim[4]; // [size, ndim, pad, pad]
  uint32_t shape_0[4];   // shape[0..3]
  uint32_t shape_1[4];   // shape[4..7]
  int32_t a_strides_0[4]; // condition strides[0..3]
  int32_t a_strides_1[4]; // condition strides[4..7]
  int32_t b_strides_0[4]; // true_val strides[0..3]
  int32_t b_strides_1[4]; // true_val strides[4..7]
  int32_t c_strides_0[4]; // false_val strides[0..3]
  int32_t c_strides_1[4]; // false_val strides[4..7]
  int32_t out_strides_0[4]; // output strides[0..3] (for general)
  int32_t out_strides_1[4]; // output strides[4..7]
};

// Generate a WGSL compute shader for the Select (ternary where) operation.
// Inputs: condition (u32, where non-zero = true), true_values, false_values.
// variant: "vvv" (all contiguous), "g" (general strided)
std::string make_select_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    const std::string& variant) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "const WORKGROUP_SIZE: u32 = 256u;\n\n";

  s << "struct TernaryParams {\n"
    << "  size_ndim: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  a_strides_0: vec4<i32>,\n"
    << "  a_strides_1: vec4<i32>,\n"
    << "  b_strides_0: vec4<i32>,\n"
    << "  b_strides_1: vec4<i32>,\n"
    << "  c_strides_0: vec4<i32>,\n"
    << "  c_strides_1: vec4<i32>,\n"
    << "  out_strides_0: vec4<i32>,\n"
    << "  out_strides_1: vec4<i32>,\n"
    << "}\n\n";

  // Condition is stored as u32 (bool-as-u32).
  s << "@group(0) @binding(0) var<storage, read> cond: array<u32>;\n"
    << "@group(0) @binding(1) var<storage, read> true_vals: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read> false_vals: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(3) var<storage, read_write> out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(4) var<uniform> params: TernaryParams;\n\n";

  if (variant == "g") {
    // General variant with strided access
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
      << "fn get_c_stride(i: u32) -> i32 {\n"
      << "  if (i < 4u) { return params.c_strides_0[i]; }\n"
      << "  return params.c_strides_1[i - 4u];\n"
      << "}\n\n"
      << "fn elem_to_loc(idx: u32, ndim: u32, get_stride: ptr<function, u32>) -> u32 {\n"
      << "  // Can't pass function pointers in WGSL, so we inline instead\n"
      << "  return 0u;\n"
      << "}\n\n"
      << "fn elem_to_loc_a(idx: u32, ndim: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_a_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n"
      << "fn elem_to_loc_b(idx: u32, ndim: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_b_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n"
      << "fn elem_to_loc_c(idx: u32, ndim: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_c_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n";
  }

  // Entry point
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x;\n"
    << "  let size = params.size_ndim.x;\n"
    << "  if (idx >= size) { return; }\n";

  if (variant == "g") {
    s << "  let ndim = params.size_ndim.y;\n"
      << "  let a_idx = elem_to_loc_a(idx, ndim);\n"
      << "  let b_idx = elem_to_loc_b(idx, ndim);\n"
      << "  let c_idx = elem_to_loc_c(idx, ndim);\n"
      << "  let condition = cond[a_idx] != 0u;\n"
      << "  let t_val = true_vals[b_idx];\n"
      << "  let f_val = false_vals[c_idx];\n";
  } else {
    // VectorVectorVector (contiguous) -- all use same index
    s << "  let condition = cond[idx] != 0u;\n"
      << "  let t_val = true_vals[idx];\n"
      << "  let f_val = false_vals[idx];\n";
  }

  s << "  out[idx] = select(f_val, t_val, condition);\n"
    << "}\n";

  return s.str();
}

} // namespace

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();

  assert(inputs.size() == 3);
  const auto& cond = inputs[0]; // condition (bool)
  const auto& b = inputs[1];    // true values
  const auto& c = inputs[2];    // false values

  auto topt = get_ternary_op_type(cond, b, c);
  set_ternary_op_output_data(cond, b, c, out, topt);

  if (out.size() == 0) {
    return;
  }

  const char* val_type = wgpu::dtype_to_wgsl(out.dtype());

  // Choose variant based on ternary op type
  std::string variant;
  if (topt == TernaryOpType::VectorVectorVector ||
      topt == TernaryOpType::ScalarScalarScalar) {
    variant = "vvv";
  } else {
    variant = "g";
  }

  std::string entry_name =
      std::string("select_") + variant + "_" + val_type;
  std::string pipeline_key = entry_name;

  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      pipeline_key,
      [&]() { return make_select_kernel(entry_name, val_type, variant); });
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(cond);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  // Fill uniform buffer
  TernaryParams params{};
  uint32_t elem_count;

  if (variant == "g") {
    auto [shape_collapsed, strides_vec] =
        collapse_contiguous_dims(cond, b, c, out);
    auto& a_strides = strides_vec[0];
    auto& b_strides = strides_vec[1];
    auto& c_strides = strides_vec[2];

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
        params.c_strides_0[i] = static_cast<int32_t>(c_strides[i]);
      } else {
        params.shape_1[i - 4] = static_cast<uint32_t>(shape_collapsed[i]);
        params.a_strides_1[i - 4] = static_cast<int32_t>(a_strides[i]);
        params.b_strides_1[i - 4] = static_cast<int32_t>(b_strides[i]);
        params.c_strides_1[i - 4] = static_cast<int32_t>(c_strides[i]);
      }
    }
  } else {
    elem_count = static_cast<uint32_t>(out.data_size());
    params.size_ndim[0] = elem_count;
  }

  WGPUBuffer uniform_buf =
      wgpu::create_uniform_buffer(&params, sizeof(TernaryParams));

  WGPUBuffer cond_buf = wgpu::wgpu_buffer(cond);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(b);
  WGPUBuffer c_buf = wgpu::wgpu_buffer(c);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pipeline,
      {{cond_buf, wgpuBufferGetSize(cond_buf)},
       {b_buf, wgpuBufferGetSize(b_buf)},
       {c_buf, wgpuBufferGetSize(c_buf)},
       {out_buf, wgpuBufferGetSize(out_buf)},
       {uniform_buf, sizeof(TernaryParams)}});

  uint32_t num_workgroups =
      (elem_count + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
}

} // namespace mlx::core
