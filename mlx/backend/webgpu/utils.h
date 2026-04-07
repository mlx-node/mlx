// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <cstring>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
#if defined(__wasm__)
  // WASI_IMPORT: JS bridge handles polling; this is a no-op.
  (void)instance;
#elif defined(WEBGPU_BACKEND_WGPU)
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
    WGPUBindGroupLayout layout,
    const std::vector<std::pair<WGPUBuffer, uint64_t>>& buffers) {
  auto& dev = device();

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
      return "u32"; // Bool stored as u32 in WebGPU
    case Dtype::Val::uint8:
      throw std::runtime_error(
          "[WebGPU] uint8 is not supported (no 8-bit storage in WGSL)");
    case Dtype::Val::uint16:
      throw std::runtime_error(
          "[WebGPU] uint16 is not supported (no 16-bit integer storage in WGSL)");
    case Dtype::Val::uint32:
      return "u32";
    case Dtype::Val::uint64:
      throw std::runtime_error(
          "[WebGPU] uint64 is not supported on WebGPU backend");
    case Dtype::Val::int8:
      throw std::runtime_error(
          "[WebGPU] int8 is not supported (no 8-bit storage in WGSL)");
    case Dtype::Val::int16:
      throw std::runtime_error(
          "[WebGPU] int16 is not supported (no 16-bit integer storage in WGSL)");
    case Dtype::Val::int32:
      return "i32";
    case Dtype::Val::int64:
      throw std::runtime_error(
          "[WebGPU] int64 is not supported on WebGPU backend");
    case Dtype::Val::float16:
      return "f16";
    case Dtype::Val::bfloat16:
      return "f32"; // Interpreted as f32; caller must ensure data is promoted
    case Dtype::Val::float32:
      return "f32";
    case Dtype::Val::float64:
      throw std::runtime_error(
          "[WebGPU] float64 is not supported on WebGPU backend");
    case Dtype::Val::complex64:
      throw std::runtime_error(
          "[WebGPU] complex64 is not supported (needs custom struct in WGSL)");
    default:
      throw std::runtime_error("[wgpu] Unsupported dtype for WGSL");
  }
}

// Emit "enable f16;\n\n" if any of the given types is "f16"
inline void emit_f16_enable(std::ostringstream& s, std::initializer_list<std::string_view> types) {
  for (auto t : types) {
    if (t == "f16") {
      s << "enable f16;\n\n";
      return;
    }
  }
}

// Emit a WGSL elem_to_loc function for strided indexing
inline void emit_elem_to_loc(
    std::ostringstream& s,
    const std::string& func_name,
    const std::string& shape_accessor,
    const std::string& stride_accessor) {
  s << "fn " << func_name << "(idx: u32, ndim: u32) -> i32 {\n"
    << "  var loc: i32 = 0;\n"
    << "  var idx_rem = idx;\n"
    << "  for (var d: u32 = ndim - 1u; d < ndim; d = d - 1u) {\n"
    << "    let s = " << shape_accessor << "(d);\n"
    << "    let st = " << stride_accessor << "(d);\n"
    << "    loc += i32(idx_rem % s) * st;\n"
    << "    idx_rem /= s;\n"
    << "  }\n"
    << "  return loc;\n"
    << "}\n\n";
}

// Fill vec4 pairs for shape and stride params
inline void fill_vec4_params(
    uint32_t* shape_0, uint32_t* shape_1,
    int32_t* strides_0, int32_t* strides_1,
    const std::vector<int32_t>& shape,
    const std::vector<int64_t>& strides,
    uint32_t ndim) {
  for (uint32_t i = 0; i < ndim && i < MAX_NDIM; ++i) {
    if (i < 4) {
      shape_0[i] = static_cast<uint32_t>(shape[i]);
      strides_0[i] = static_cast<int32_t>(strides[i]);
    } else {
      shape_1[i - 4] = static_cast<uint32_t>(shape[i]);
      strides_1[i - 4] = static_cast<int32_t>(strides[i]);
    }
  }
}

} // namespace mlx::core::wgpu
