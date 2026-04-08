// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <mutex>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

namespace mlx::core::wgpu {

class CommandEncoder;

using allocator::Buffer;

// Stores a WebGPU buffer with optional CPU-resident data for readback.
struct WebGPUBuffer {
  WGPUBuffer buffer;
  size_t size;
  void* cpu_ptr{nullptr}; // Non-null after raw_ptr() readback
  bool cpu_dirty{false};   // True when cpu_ptr has data not yet uploaded to GPU
  bool gpu_has_data{false}; // True when GPU buffer has meaningful data
};

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
