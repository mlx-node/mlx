// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <mutex>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/dtype.h"

namespace mlx::core {
class array;
}

namespace mlx::core::wgpu {

class CommandEncoder;

using allocator::Buffer;

// How the GPU buffer stores its data relative to the logical CPU dtype.
//   Upconverted: bf16 is expanded to f32 on upload (today's baseline).
//   PackedBf16:  bf16 is stored raw, 2 elements per u32 on the GPU. The
//                kernel is responsible for unpacking on the fly.
enum class StorageMode : uint8_t {
  Upconverted,
  PackedBf16,
};

// Stores a WebGPU buffer with optional CPU-resident data for readback.
// The dtype_val field tracks the array's dtype for upload/download format
// conversion between CPU types (bool=1B, bf16=2B) and GPU types (u32=4B, f32=4B).
struct WebGPUBuffer {
  WGPUBuffer buffer;
  size_t size;
  void* cpu_ptr{nullptr};  // Non-null after raw_ptr() readback
  // Byte size of the CPU-side allocation at `cpu_ptr`. Captured when the
  // allocation is made so that `upload_with_conversion` can always upload
  // the whole owning buffer, even if the view that triggered the upload
  // only references a slice of it. Without this, a slice-upload would only
  // copy the first N/2 elements, leaving the slice's actual region zeroed.
  size_t cpu_bytes{0};
  bool cpu_dirty{false};    // True when cpu_ptr has data not yet uploaded to GPU
  bool gpu_has_data{false};  // True when GPU buffer has meaningful data
  Dtype::Val dtype_val{Dtype::Val::float32}; // For upload/download conversion
  StorageMode storage_mode{StorageMode::Upconverted};
};

// Runtime opt-in for the packed-bf16 upload / GEMV fast path.
// When enabled, eligible bf16 weight buffers (>= 4096 elements) are kept
// packed 2-per-u32 on the GPU and a matching kernel variant is selected
// in matmul. Defaults to false; plumbed from the TS init message.
void set_packed_bf16_enabled(bool enabled);
bool packed_bf16_enabled();

// Runtime force-on for the SDPA decomposed fallback. When true, the SDPA
// primitive reports `use_fallback() == true` unconditionally so the op
// decomposes into matmul(Q·K^T) + softmax + matmul(·V). Used by the demo
// app to A/B test the fused vector + tile kernels against the baseline
// without a rebuild. Defaults to false; plumbed from ?sdpa_fallback=1.
void set_sdpa_fallback_forced(bool enabled);
bool sdpa_fallback_forced();

// Helpers that flip an array's WebGPUBuffer into StorageMode::PackedBf16.
// These are defined in allocator.cpp (which has the full WebGPUBuffer layout
// + mlx::core::array include path) so that the FFI-bridge translation unit
// in mlx-sys can call them without pulling in <webgpu/webgpu.h>.
//
// try_opt_in_packed_bf16: opts in only if the runtime flag is enabled, the
//   dtype is bf16, size >= min_elements, and the buffer is still in the
//   "upload-pending" state (never on a post-GPU-eval buffer). Returns true
//   if the flip happened.
// mark_buffer_packed_bf16: unconditionally flips the bookkeeping flag.
//   Caller must have already placed packed u32 pairs in the underlying
//   WGPUBuffer (e.g. the JS-side weight-upload path in gpu-worker.ts).
bool try_opt_in_packed_bf16(mlx::core::array& arr, size_t min_elements);
bool mark_buffer_packed_bf16(mlx::core::array& arr);

class WebGPUAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  WebGPUAllocator();
  friend WebGPUAllocator& allocator();

  void free_wgpu_buffer(WebGPUBuffer* buf);

  std::mutex mutex_;
  size_t memory_limit_;
  size_t max_pool_size_;
  BufferCache<WebGPUBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
};

WebGPUAllocator& allocator();

} // namespace mlx::core::wgpu
