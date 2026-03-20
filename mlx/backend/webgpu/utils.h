// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/dtype.h"

namespace mlx::core::wgpu {

// Shared constants used across WebGPU kernel files.
constexpr uint32_t WORKGROUP_SIZE = 256;
constexpr uint32_t MAX_NDIM = 8;
constexpr uint32_t N_READS = 4;

// Poll the WebGPU instance to process pending async operations.
inline void poll_instance(WGPUInstance instance) {
#if defined(WEBGPU_BACKEND_WGPU)
  wgpuInstancePoll(instance, false, nullptr);
#else
  wgpuInstanceProcessEvents(instance);
#endif
}

// Create a WGPUBuffer with initial data for the given usage flags.
inline WGPUBuffer create_buffer_with_data(
    const void* data,
    size_t byte_size,
    WGPUBufferUsageFlags usage) {
  auto& dev = device();
  WGPUBufferDescriptor desc = {};
  desc.size = byte_size;
  desc.usage = usage;
  desc.mappedAtCreation = true;

  WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &desc);
  if (!buf) {
    throw std::runtime_error("[WebGPU] Failed to create buffer");
  }
  void* mapped = wgpuBufferGetMappedRange(buf, 0, byte_size);
  if (!mapped) {
    wgpuBufferRelease(buf);
    throw std::runtime_error("[WebGPU] Failed to get mapped range");
  }
  std::memcpy(mapped, data, byte_size);
  wgpuBufferUnmap(buf);
  return buf;
}

// Create a uniform buffer with initial data.
inline WGPUBuffer create_uniform_buffer(const void* data, size_t byte_size) {
  return create_buffer_with_data(
      data, byte_size, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
}

// Create a storage buffer with initial data.
inline WGPUBuffer create_storage_buffer(const void* data, size_t byte_size) {
  return create_buffer_with_data(
      data, byte_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
}

// Create a bind group with a dynamic number of buffer entries.
// Each pair is (buffer, size). Bindings are assigned sequentially from 0.
inline WGPUBindGroup create_bind_group(
    WGPUComputePipeline pipeline,
    const std::vector<std::pair<WGPUBuffer, uint64_t>>& buffers) {
  auto& dev = device();
  WGPUBindGroupLayout layout =
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
  if (!layout) {
    throw std::runtime_error("[WebGPU] Failed to get bind group layout");
  }

  std::vector<WGPUBindGroupEntry> entries(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    entries[i] = {};
    entries[i].binding = static_cast<uint32_t>(i);
    entries[i].buffer = buffers[i].first;
    entries[i].offset = 0;
    entries[i].size = buffers[i].second;
  }

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = layout;
  bg_desc.entryCount = static_cast<uint32_t>(entries.size());
  bg_desc.entries = entries.data();

  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev.gpu_device(), &bg_desc);
  wgpuBindGroupLayoutRelease(layout);

  if (!bg) {
    throw std::runtime_error("[WebGPU] Failed to create bind group");
  }
  return bg;
}

// Extract the WGPUBuffer from an array's allocator buffer.
inline WGPUBuffer wgpu_buffer(const array& arr) {
  auto* buf = static_cast<const WebGPUBuffer*>(arr.buffer().ptr());
  return buf->buffer;
}

// Compute the byte offset into the array's buffer.
inline uint64_t wgpu_offset(const array& arr) {
  return static_cast<uint64_t>(arr.offset());
}

// Compute the data size in bytes for an array.
inline uint64_t wgpu_data_size(const array& arr) {
  return static_cast<uint64_t>(arr.data_size()) * arr.itemsize();
}

// Map MLX dtype to WGSL type name string.
inline const char* dtype_to_wgsl(Dtype dtype) {
  switch (dtype.val()) {
    case Dtype::Val::bool_:
      return "bool";
    case Dtype::Val::uint8:
      return "u32"; // WebGPU lacks 8-bit; pack into u32
    case Dtype::Val::uint16:
      return "u32"; // WebGPU lacks 16-bit; pack into u32
    case Dtype::Val::uint32:
      return "u32";
    case Dtype::Val::uint64:
      return "u32"; // No native u64 in WGSL
    case Dtype::Val::int8:
      return "i32"; // WebGPU lacks 8-bit
    case Dtype::Val::int16:
      return "i32"; // WebGPU lacks 16-bit
    case Dtype::Val::int32:
      return "i32";
    case Dtype::Val::int64:
      return "i32"; // No native i64 in WGSL
    case Dtype::Val::float16:
      return "f16";
    case Dtype::Val::bfloat16:
      return "f32"; // bfloat16 needs software emulation
    case Dtype::Val::float32:
      return "f32";
    case Dtype::Val::float64:
      return "f32"; // No native f64 in WGSL
    case Dtype::Val::complex64:
      return "f32"; // complex needs custom struct
    default:
      throw std::runtime_error("[wgpu] Unsupported dtype for WGSL");
  }
}

} // namespace mlx::core::wgpu
