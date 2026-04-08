// Copyright 2026 Apple Inc.
//
// WebGPU Convolution implementation.
//
// Currently supports depthwise conv1d (groups == in_channels):
//   Input:  [B, T, C]
//   Weight: [C, K, 1]  (one kernel per channel)
//   Output: [B, T_out, C]
//
// Each thread computes one output element by iterating over the kernel window.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// ---------------------------------------------------------------------------
// Uniform params for the Conv1D kernel (vec4-aligned).
// ---------------------------------------------------------------------------

struct Conv1DParams {
  uint32_t data[4];  // [out_size, in_channels, kernel_size, in_length]
  uint32_t data2[4]; // [out_length, groups, stride, batch_size]
  uint32_t data3[4]; // [padding_lo, dilation, flip, 0]
};

// ---------------------------------------------------------------------------
// WGSL kernel generation for depthwise conv1d
// ---------------------------------------------------------------------------

std::string make_conv1d_depthwise_kernel(
    const std::string& entry_name,
    const std::string& type) {
  std::ostringstream s;

  if (type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct Conv1DParams {\n"
    << "  data: vec4<u32>,\n"
    << "  data2: vec4<u32>,\n"
    << "  data3: vec4<u32>,\n"
    << "}\n\n";

  // Bindings: input, weight, output, params
  s << "@group(0) @binding(0) var<storage, read> input: array<" << type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read> weight: array<" << type
    << ">;\n"
    << "@group(0) @binding(2) var<storage, read_write> output: array<" << type
    << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: Conv1DParams;\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let out_size = params.data.x;\n"
    << "  let in_channels = params.data.y;\n"
    << "  let kernel_size = params.data.z;\n"
    << "  let in_length = params.data.w;\n"
    << "  let out_length = params.data2.x;\n"
    << "  // params.data2.y = groups (unused in depthwise — equals in_channels)\n"
    << "  let stride = params.data2.z;\n"
    << "  let batch_size = params.data2.w;\n"
    << "  let padding_lo = params.data3.x;\n"
    << "  let dilation = params.data3.y;\n"
    << "  let flip = params.data3.z;\n"
    << "\n"
    << "  let out_idx = gid.x;\n"
    << "  if (out_idx >= out_size) { return; }\n"
    << "\n"
    << "  // Decompose flat output index -> [batch, out_t, channel]\n"
    << "  let channel = out_idx % in_channels;\n"
    << "  let rem = out_idx / in_channels;\n"
    << "  let out_t = rem % out_length;\n"
    << "  let batch = rem / out_length;\n"
    << "\n"
    << "  var sum: " << type << " = " << type << "(0.0);\n"
    << "  for (var k: u32 = 0u; k < kernel_size; k = k + 1u) {\n"
    << "    // Compute the input time index for this kernel position\n"
    << "    let in_t_signed: i32 = i32(out_t * stride) - i32(padding_lo) + i32(k * dilation);\n"
    << "    if (in_t_signed >= 0 && in_t_signed < i32(in_length)) {\n"
    << "      let in_t = u32(in_t_signed);\n"
    << "      // Input layout: [B, T, C] -> index = batch * in_length * in_channels + in_t * in_channels + channel\n"
    << "      let in_idx = batch * in_length * in_channels + in_t * in_channels + channel;\n"
    << "      // Weight layout: [C, K, 1] -> index = channel * kernel_size + k_idx\n"
    << "      var k_idx = k;\n"
    << "      if (flip != 0u) { k_idx = kernel_size - 1u - k; }\n"
    << "      let w_idx = channel * kernel_size + k_idx;\n"
    << "      sum = sum + input[in_idx] * weight[w_idx];\n"
    << "    }\n"
    << "  }\n"
    << "\n"
    << "  // Output layout: [B, T_out, C]\n"
    << "  output[out_idx] = sum;\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL kernel generation for general grouped conv1d
// ---------------------------------------------------------------------------

std::string make_conv1d_general_kernel(
    const std::string& entry_name,
    const std::string& type) {
  std::ostringstream s;

  if (type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct Conv1DParams {\n"
    << "  data: vec4<u32>,\n"
    << "  data2: vec4<u32>,\n"
    << "  data3: vec4<u32>,\n"
    << "}\n\n";

  // Bindings: input, weight, output, params
  s << "@group(0) @binding(0) var<storage, read> input: array<" << type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read> weight: array<" << type
    << ">;\n"
    << "@group(0) @binding(2) var<storage, read_write> output: array<" << type
    << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: Conv1DParams;\n\n";

  // General grouped conv1d:
  //   Input:  [B, T, C_in]
  //   Weight: [C_out, K, C_in/groups]
  //   Output: [B, T_out, C_out]
  //
  // For each output element (batch, out_t, c_out):
  //   group = c_out / (C_out / groups)
  //   c_in_start = group * (C_in / groups)
  //   sum over k in [0, K), c in [0, C_in/groups):
  //     input[batch, out_t*stride - pad + k*dilation, c_in_start + c]
  //     * weight[c_out, k, c]

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let out_size = params.data.x;\n"
    << "  let in_channels = params.data.y;\n"
    << "  let kernel_size = params.data.z;\n"
    << "  let in_length = params.data.w;\n"
    << "  let out_length = params.data2.x;\n"
    << "  let groups = params.data2.y;\n"
    << "  let stride = params.data2.z;\n"
    << "  let batch_size = params.data2.w;\n"
    << "  let padding_lo = params.data3.x;\n"
    << "  let dilation = params.data3.y;\n"
    << "  let flip = params.data3.z;\n"
    << "\n"
    << "  let out_idx = gid.x;\n"
    << "  if (out_idx >= out_size) { return; }\n"
    << "\n"
    << "  // Output shape: [B, T_out, C_out]\n"
    << "  // We need C_out from output, which is out_size / (batch_size * out_length)\n"
    << "  let out_channels = out_size / (batch_size * out_length);\n"
    << "  let c_out = out_idx % out_channels;\n"
    << "  let rem = out_idx / out_channels;\n"
    << "  let out_t = rem % out_length;\n"
    << "  let batch = rem / out_length;\n"
    << "\n"
    << "  let channels_per_group = in_channels / groups;\n"
    << "  let out_channels_per_group = out_channels / groups;\n"
    << "  let group = c_out / out_channels_per_group;\n"
    << "  let c_in_start = group * channels_per_group;\n"
    << "\n"
    << "  var sum: " << type << " = " << type << "(0.0);\n"
    << "  for (var k: u32 = 0u; k < kernel_size; k = k + 1u) {\n"
    << "    let in_t_signed: i32 = i32(out_t * stride) - i32(padding_lo) + i32(k * dilation);\n"
    << "    if (in_t_signed >= 0 && in_t_signed < i32(in_length)) {\n"
    << "      let in_t = u32(in_t_signed);\n"
    << "      var k_idx = k;\n"
    << "      if (flip != 0u) { k_idx = kernel_size - 1u - k; }\n"
    << "      for (var c: u32 = 0u; c < channels_per_group; c = c + 1u) {\n"
    << "        // Input: [B, T, C_in]\n"
    << "        let in_idx = batch * in_length * in_channels + in_t * in_channels + c_in_start + c;\n"
    << "        // Weight: [C_out, K, C_in/groups]\n"
    << "        let w_idx = c_out * kernel_size * channels_per_group + k_idx * channels_per_group + c;\n"
    << "        sum = sum + input[in_idx] * weight[w_idx];\n"
    << "      }\n"
    << "    }\n"
    << "  }\n"
    << "\n"
    << "  output[out_idx] = sum;\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point: Convolution::eval_gpu
// ---------------------------------------------------------------------------

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& s = stream();

  const auto& in_orig = inputs[0];
  const auto& wt_orig = inputs[1];

  // Only support 1D convolution for now (input is 3D: [B, T, C])
  if (in_orig.ndim() != 3) {
    throw std::runtime_error(
        "[Convolution::eval_gpu] WebGPU backend only supports conv1d "
        "(3D input) for now.");
  }

  // Ensure inputs are contiguous
  array in = in_orig;
  auto& encoder = wgpu::get_command_encoder(s);
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in_orig, s);
    encoder.add_temporary(in);
  }

  array wt = wt_orig;
  if (!wt.flags().row_contiguous) {
    wt = contiguous_copy_gpu(wt_orig, s);
    encoder.add_temporary(wt);
  }

  // Allocate output
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  if (out.size() == 0) {
    return;
  }

  auto& dev = wgpu::device();

  // Extract conv parameters
  uint32_t batch_size = static_cast<uint32_t>(in.shape(0));
  uint32_t in_length = static_cast<uint32_t>(in.shape(1));
  uint32_t in_channels = static_cast<uint32_t>(in.shape(2));
  uint32_t kernel_size = static_cast<uint32_t>(wt.shape(1));
  uint32_t groups = static_cast<uint32_t>(groups_);
  uint32_t stride_val = static_cast<uint32_t>(kernel_strides_[0]);
  uint32_t padding_lo = static_cast<uint32_t>(padding_lo_[0]);
  uint32_t dilation = static_cast<uint32_t>(kernel_dilation_[0]);
  uint32_t flip_val = flip_ ? 1u : 0u;
  uint32_t out_length = static_cast<uint32_t>(out.shape(1));
  uint32_t out_size = static_cast<uint32_t>(out.size());

  bool is_depthwise = (groups == in_channels);

  // Get WGSL type
  const char* wgsl_type = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string type(wgsl_type);

  // Build kernel name and select kernel generator
  std::string entry_name;
  if (is_depthwise) {
    entry_name = std::string("conv1d_depthwise_") + type;
  } else {
    entry_name = std::string("conv1d_general_") + type;
  }

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name, [&]() {
        if (is_depthwise) {
          return make_conv1d_depthwise_kernel(entry_name, type);
        } else {
          return make_conv1d_general_kernel(entry_name, type);
        }
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_input_array(wt);
  encoder.set_output_array(out);

  // Build params
  Conv1DParams params{};
  params.data[0] = out_size;
  params.data[1] = in_channels;
  params.data[2] = kernel_size;
  params.data[3] = in_length;
  params.data2[0] = out_length;
  params.data2[1] = groups;
  params.data2[2] = stride_val;
  params.data2[3] = batch_size;
  params.data3[0] = padding_lo;
  params.data3[1] = dilation;
  params.data3[2] = flip_val;
  params.data3[3] = 0;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(Conv1DParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer wt_buf = wgpu::wgpu_buffer(wt);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t wt_buf_size = wgpuBufferGetSize(wt_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {wt_buf, wt_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(Conv1DParams)}});

  // Dispatch one thread per output element
  uint32_t num_workgroups =
      (out_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core
