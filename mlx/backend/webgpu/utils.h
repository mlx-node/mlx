// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/dtype.h"

namespace mlx::core::wgpu {

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
