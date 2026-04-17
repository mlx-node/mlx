// Copyright 2026 Apple Inc.
//
// WebGPU ArgReduce operations (argmax, argmin).
//
// Simple per-output-element kernel: each thread computes one output element
// by iterating over the full reduction axis and tracking the index of
// the max/min value.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>

namespace mlx::core {

namespace {

// ---------------------------------------------------------------------------
// Uniform params for the ArgReduce kernel (vec4-aligned).
// ---------------------------------------------------------------------------

struct ArgReduceParams {
  uint32_t data[4];       // [out_size, axis_size, axis_stride, ndim]
  uint32_t shape_0[4];    // shape[0..3]  (with axis removed)
  uint32_t shape_1[4];    // shape[4..7]
  int32_t in_strides_0[4];  // input strides[0..3] (with axis removed)
  int32_t in_strides_1[4];  // input strides[4..7]
};

// ---------------------------------------------------------------------------
// WGSL kernel generation
// ---------------------------------------------------------------------------

std::string make_arg_reduce_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    bool is_argmax) {
  std::ostringstream s;

  if (in_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct ArgReduceParams {\n"
    << "  data: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  in_strides_0: vec4<i32>,\n"
    << "  in_strides_1: vec4<i32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<u32>;\n"
    << "@group(0) @binding(2) var<uniform> params: ArgReduceParams;\n\n";

  // Helper functions for shape/stride access
  s << "fn get_shape(i: u32) -> u32 {\n"
    << "  if (i < 4u) { return params.shape_0[i]; }\n"
    << "  return params.shape_1[i - 4u];\n"
    << "}\n\n"
    << "fn get_in_stride(i: u32) -> i32 {\n"
    << "  if (i < 4u) { return params.in_strides_0[i]; }\n"
    << "  return params.in_strides_1[i - 4u];\n"
    << "}\n\n";

  // elem_to_loc: compute input offset from output linear index
  s << "fn elem_to_loc(idx: u32, ndim: u32) -> u32 {\n"
    << "  var loc: i32 = 0;\n"
    << "  var idx_rem: u32 = idx;\n"
    << "  for (var d: u32 = ndim - 1u; d < ndim; d = d - 1u) {\n"
    << "    let dim_size = get_shape(d);\n"
    << "    loc += i32(idx_rem % dim_size) * get_in_stride(d);\n"
    << "    idx_rem = idx_rem / dim_size;\n"
    << "  }\n"
    << "  return u32(loc);\n"
    << "}\n\n";

  // The comparison operator
  const char* cmp_op = is_argmax ? ">" : "<";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let out_size = params.data.x;\n"
    << "  let axis_size = params.data.y;\n"
    << "  let axis_stride = params.data.z;\n"
    << "  let ndim = params.data.w;\n"
    << "\n"
    << "  let out_idx = gid.x;\n"
    << "  if (out_idx >= out_size) { return; }\n"
    << "\n";

  // Compute the base input offset from the output index
  s << "  var base_offset: u32 = 0u;\n"
    << "  if (ndim > 0u) {\n"
    << "    base_offset = elem_to_loc(out_idx, ndim);\n"
    << "  }\n"
    << "\n";

  // Iterate over the reduction axis to find the argmax/argmin
  s << "  var best_val: " << in_type << " = input[base_offset];\n"
    << "  var best_idx: u32 = 0u;\n"
    << "  for (var i: u32 = 1u; i < axis_size; i = i + 1u) {\n"
    << "    let val = input[base_offset + i * axis_stride];\n"
    << "    if (val " << cmp_op << " best_val) {\n"
    << "      best_val = val;\n"
    << "      best_idx = i;\n"
    << "    }\n"
    << "  }\n"
    << "\n"
    << "  output[out_idx] = best_idx;\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point: ArgReduce::eval_gpu
// ---------------------------------------------------------------------------

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // Output is always uint32 (indices)
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  if (in.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  bool is_argmax = (reduce_type_ == ArgReduce::ArgMax);

  // Get the WGSL type for the input
  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string in_type = (std::string(in_wgsl) == "bool") ? "u32" : in_wgsl;

  // Prepare shape/strides with the reduction axis removed
  auto shape = in.shape();
  auto in_strides = in.strides();
  auto axis_size = static_cast<uint32_t>(shape[axis_]);
  auto axis_stride = static_cast<uint32_t>(in_strides[axis_]);

  // Remove the reduction axis from shape and strides
  shape.erase(shape.begin() + axis_);
  in_strides.erase(in_strides.begin() + axis_);
  uint32_t ndim = static_cast<uint32_t>(shape.size());

  // Build kernel name
  std::string op_name = is_argmax ? "argmax" : "argmin";
  std::string entry_name = op_name + "_" + in_type;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_arg_reduce_kernel(entry_name, in_type, is_argmax); });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Build params
  ArgReduceParams params{};
  params.data[0] = static_cast<uint32_t>(out.size());
  params.data[1] = axis_size;
  params.data[2] = axis_stride;
  params.data[3] = ndim;

  for (uint32_t i = 0; i < ndim && i < wgpu::MAX_NDIM; ++i) {
    if (i < 4) {
      params.shape_0[i] = static_cast<uint32_t>(shape[i]);
      params.in_strides_0[i] = static_cast<int32_t>(in_strides[i]);
    } else {
      params.shape_1[i - 4] = static_cast<uint32_t>(shape[i]);
      params.in_strides_1[i - 4] = static_cast<int32_t>(in_strides[i]);
    }
  }

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(ArgReduceParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(ArgReduceParams)}});

  // Dispatch one thread per output element
  uint32_t num_workgroups =
      (static_cast<uint32_t>(out.size()) + wgpu::WORKGROUP_SIZE - 1) /
      wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core
