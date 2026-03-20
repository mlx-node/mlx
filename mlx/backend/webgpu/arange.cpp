// Copyright 2026 Apple Inc.
//
// WebGPU Arange implementation.
//
// Generates a WGSL kernel that fills an output buffer with
// start + step * idx for each element index.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mlx::core {

namespace {

constexpr uint32_t WORKGROUP_SIZE = 256;

// Uniform buffer layout for Arange kernel.
struct ArangeParams {
  uint32_t size;
  float start;
  float step;
  uint32_t _pad;
};

WGPUBuffer create_uniform_buffer(const void* data, size_t byte_size) {
  auto& dev = wgpu::device();
  WGPUBufferDescriptor desc = {};
  desc.size = byte_size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;

  WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &desc);
  if (!buf) {
    throw std::runtime_error(
        "[WebGPU arange] Failed to create uniform buffer");
  }
  void* mapped = wgpuBufferGetMappedRange(buf, 0, byte_size);
  std::memcpy(mapped, data, byte_size);
  wgpuBufferUnmap(buf);
  return buf;
}

WGPUShaderModule
get_shader_module(const std::string& key, const std::string& source) {
  static std::mutex mtx;
  static std::unordered_map<std::string, WGPUShaderModule> cache;

  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  auto& dev = wgpu::device();

  WGPUShaderModuleWGSLDescriptor wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgsl_desc.code = source.c_str();

  WGPUShaderModuleDescriptor desc = {};
  desc.nextInChain = &wgsl_desc.chain;
  desc.label = key.c_str();

  WGPUShaderModule mod =
      wgpuDeviceCreateShaderModule(dev.gpu_device(), &desc);
  if (!mod) {
    throw std::runtime_error(
        "[WebGPU arange] Failed to create shader module: " + key);
  }

  cache[key] = mod;
  return mod;
}

// Generate WGSL kernel for arange.
// out[idx] = out_type(start + step * f32(idx))
std::string make_arange_kernel(
    const std::string& entry_name,
    const std::string& out_type) {
  std::ostringstream s;

  if (out_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "struct ArangeParams {\n"
    << "  size: u32,\n"
    << "  start: f32,\n"
    << "  step: f32,\n"
    << "  _pad: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read_write> out: array<"
    << out_type << ">;\n"
    << "@group(0) @binding(1) var<uniform> params: ArangeParams;\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x;\n"
    << "  if (idx >= params.size) { return; }\n"
    << "  let val = params.start + params.step * f32(idx);\n"
    << "  out[idx] = " << out_type << "(val);\n"
    << "}\n";

  return s.str();
}

} // namespace

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  if (out.size() == 0) {
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  const char* out_type = wgpu::dtype_to_wgsl(out.dtype());

  std::string entry_name = std::string("arange_") + out_type;
  std::string pipeline_key = entry_name;

  std::string source = make_arange_kernel(entry_name, out_type);
  WGPUShaderModule shader = get_shader_module(pipeline_key, source);

  auto& dev = wgpu::device();
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_output_array(out);

  // Fill params
  ArangeParams params{};
  params.size = static_cast<uint32_t>(out.data_size());
  params.start = static_cast<float>(start_);
  params.step = static_cast<float>(step_);

  WGPUBuffer uniform_buf =
      create_uniform_buffer(&params, sizeof(ArangeParams));

  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  // Bind group: output (binding 0), params (binding 1)
  WGPUBindGroupLayout layout =
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

  WGPUBindGroupEntry entries[2] = {};
  entries[0].binding = 0;
  entries[0].buffer = out_buf;
  entries[0].offset = 0;
  entries[0].size = out_buf_size;

  entries[1].binding = 1;
  entries[1].buffer = uniform_buf;
  entries[1].offset = 0;
  entries[1].size = sizeof(ArangeParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = layout;
  bg_desc.entryCount = 2;
  bg_desc.entries = entries;

  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev.gpu_device(), &bg_desc);
  wgpuBindGroupLayoutRelease(layout);

  if (!bg) {
    throw std::runtime_error("[WebGPU arange] Failed to create bind group");
  }

  uint32_t num_workgroups =
      (params.size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  encoder.dispatch_compute(pipeline, {bg}, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
}

} // namespace mlx::core
