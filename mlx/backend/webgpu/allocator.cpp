// Copyright 2026 Apple Inc.

#include "mlx/backend/webgpu/allocator.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <cassert>
#include <condition_variable>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace wgpu {

constexpr size_t page_size = 16384;

WebGPUAllocator::WebGPUAllocator()
    : buffer_cache_(
          page_size,
          [](WebGPUBuffer* buf) { return buf->size; },
          [this](WebGPUBuffer* buf) { free_wgpu_buffer(buf); }),
      memory_limit_(1ULL << 30), // 1 GB (fits in WASM32 size_t)
      max_pool_size_(1ULL << 29) // 512 MB
{}

Buffer WebGPUAllocator::malloc(size_t size) {
  if (size == 0) {
    return Buffer{new WebGPUBuffer{nullptr, 0, nullptr}};
  }

  // Round up size
  if (size <= 8) {
    size = 8;
  } else if (size < page_size) {
    size = next_power_of_2(size);
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  // Try to find a buffer in the cache
  std::unique_lock lock(mutex_);
  WebGPUBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (buf) {
    // Reset state for reused buffer
    if (buf->cpu_ptr) {
      std::free(buf->cpu_ptr);
      buf->cpu_ptr = nullptr;
    }
    buf->cpu_dirty = false;
    buf->gpu_has_data = false;
  }
  if (!buf) {
    // Release cached buffers under memory pressure
    int64_t mem_to_free =
        static_cast<int64_t>(get_active_memory() + get_cache_memory() + size) -
        static_cast<int64_t>(memory_limit_);
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(mem_to_free);
    }
    lock.unlock();

    // Create a new WGPUBuffer
    auto& dev = device();
    if (!dev.is_valid()) {
      throw std::runtime_error(
          "[WebGPU] Cannot allocate buffer: device not initialized");
    }

    WGPUBufferDescriptor buf_desc = {};
    buf_desc.size = size;
    buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
        WGPUBufferUsage_CopySrc;
    buf_desc.mappedAtCreation = false;

    WGPUBuffer gpu_buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &buf_desc);
    if (!gpu_buf) {
      std::ostringstream msg;
      msg << "[WebGPU] Unable to allocate " << size << " bytes.";
      throw std::runtime_error(msg.str());
    }

    buf = new WebGPUBuffer{gpu_buf, size, nullptr, false, false};
    lock.lock();
  }

  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain cache below the requested limit
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }

  return Buffer{buf};
}

void WebGPUAllocator::free(Buffer buffer) {
  auto* buf = static_cast<WebGPUBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  if (buf->size == 0) {
    delete buf;
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;

  // If it has been read back to CPU, don't cache it
  if (buf->cpu_ptr) {
    lock.unlock();
    free_wgpu_buffer(buf);
    return;
  }

  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    free_wgpu_buffer(buf);
  }
}

size_t WebGPUAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<WebGPUBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

void WebGPUAllocator::free_wgpu_buffer(WebGPUBuffer* buf) {
  if (buf->cpu_ptr) {
    std::free(buf->cpu_ptr);
    buf->cpu_ptr = nullptr;
  }
  if (buf->buffer) {
    wgpuBufferDestroy(buf->buffer);
    wgpuBufferRelease(buf->buffer);
  }
  delete buf;
}

size_t WebGPUAllocator::get_active_memory() const {
  return active_memory_;
}

size_t WebGPUAllocator::get_peak_memory() const {
  return peak_memory_;
}

void WebGPUAllocator::reset_peak_memory() {
  std::lock_guard lock(mutex_);
  peak_memory_ = 0;
}

size_t WebGPUAllocator::get_memory_limit() {
  return memory_limit_;
}

size_t WebGPUAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  std::swap(limit, memory_limit_);
  return limit;
}

size_t WebGPUAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t WebGPUAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

void WebGPUAllocator::clear_cache() {
  std::lock_guard lock(mutex_);
  buffer_cache_.clear();
}

WebGPUAllocator& allocator() {
  static auto* allocator_ = []() {
    // Ensure scheduler is created before allocator.
    scheduler::scheduler();
    // Heap-allocated, intentionally leaked (same as CUDA pattern).
    return new WebGPUAllocator();
  }();
  return *allocator_;
}

} // namespace wgpu

namespace allocator {

Allocator& allocator() {
  return wgpu::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }

  auto& wbuf = *static_cast<wgpu::WebGPUBuffer*>(ptr_);

  // Already read back to CPU
  if (wbuf.cpu_ptr) {
    return wbuf.cpu_ptr;
  }

  if (!wbuf.buffer || wbuf.size == 0) {
    return nullptr;
  }

  // If the GPU buffer has no meaningful data (freshly allocated, never
  // computed on), skip the expensive GPU readback and just allocate
  // CPU memory. The caller (array::init) will write data into this
  // pointer, and we mark it dirty so it gets uploaded to GPU before
  // any GPU compute reads from it.
  if (!wbuf.gpu_has_data) {
    void* cpu_data = std::malloc(wbuf.size);
    if (!cpu_data) {
      throw std::runtime_error("[WebGPU] Failed to allocate CPU memory");
    }
    std::memset(cpu_data, 0, wbuf.size);
    wbuf.cpu_ptr = cpu_data;
    wbuf.cpu_dirty = true;
    return cpu_data;
  }

  // Flush all pending command encoders to ensure GPU compute work
  // has been submitted to the queue before we issue the copy.
  // Without this, the readback copy can execute before the compute
  // that produced the data.
  wgpu::flush_all_encoders();

  auto& dev = wgpu::device();

  // Create staging buffer with MapRead usage
  WGPUBufferDescriptor staging_desc = {};
  staging_desc.size = wbuf.size;
  staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
  staging_desc.mappedAtCreation = false;

  WGPUBuffer staging = wgpuDeviceCreateBuffer(dev.gpu_device(), &staging_desc);
  if (!staging) {
    throw std::runtime_error(
        "[WebGPU] Failed to create staging buffer for readback");
  }

  // Encode copy: GPU buffer -> staging buffer
  WGPUCommandEncoderDescriptor enc_desc = {};
  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(dev.gpu_device(), &enc_desc);
  if (!encoder) {
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
    throw std::runtime_error(
        "[WebGPU] Failed to create command encoder for buffer readback");
  }
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder, wbuf.buffer, 0, staging, 0, wbuf.size);
  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(encoder, nullptr);
  wgpuQueueSubmit(dev.gpu_queue(), 1, &cmd_buf);
  wgpuCommandBufferRelease(cmd_buf);
  wgpuCommandEncoderRelease(encoder);

  // Map the staging buffer (blocking)
  struct MapData {
    bool done = false;
    WGPUBufferMapAsyncStatus status;
    std::mutex mtx;
    std::condition_variable cv;
  };
  MapData map_data;

  wgpuBufferMapAsync(
      staging,
      WGPUMapMode_Read,
      0,
      wbuf.size,
      [](WGPUBufferMapAsyncStatus status, void* userdata) {
        auto* data = static_cast<MapData*>(userdata);
        {
          std::lock_guard<std::mutex> lock(data->mtx);
          data->status = status;
          data->done = true;
        }
        data->cv.notify_one();
      },
      &map_data);

  // Poll until mapping completes
  {
    std::unique_lock<std::mutex> lock(map_data.mtx);
    while (!map_data.done) {
      lock.unlock();
      wgpu::poll_instance(dev.gpu_instance());
      lock.lock();
    }
  }

  if (map_data.status != WGPUBufferMapAsyncStatus_Success) {
    wgpuBufferRelease(staging);
    throw std::runtime_error("[WebGPU] Failed to map staging buffer");
  }

  // Copy mapped data to CPU allocation
  const void* mapped =
      wgpuBufferGetConstMappedRange(staging, 0, wbuf.size);
  if (!mapped) {
    wgpuBufferUnmap(staging);
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
    throw std::runtime_error(
        "[WebGPU] Failed to get mapped range for staging buffer readback");
  }
  void* cpu_data = std::malloc(wbuf.size);
  if (!cpu_data) {
    wgpuBufferUnmap(staging);
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
    throw std::runtime_error("[WebGPU] Failed to allocate CPU memory");
  }
  std::memcpy(cpu_data, mapped, wbuf.size);

  // Convert from GPU format to CPU format for types with wider GPU representation
  if (wbuf.dtype_val == Dtype::Val::bfloat16) {
    // Contract f32 (4 bytes) -> bf16 (2 bytes) in-place (forward safe: 4B->2B)
    size_t n = wbuf.size / 4;
    auto* src = static_cast<const float*>(cpu_data);
    auto* dst = static_cast<bfloat16_t*>(cpu_data);
    for (size_t i = 0; i < n; i++) {
      dst[i] = static_cast<bfloat16_t>(src[i]);
    }
  } else if (wbuf.dtype_val == Dtype::Val::bool_) {
    // Contract u32 (4 bytes) -> bool (1 byte) in-place (forward safe: 4B->1B)
    size_t n = wbuf.size / 4;
    auto* src = static_cast<const uint32_t*>(cpu_data);
    auto* dst = static_cast<uint8_t*>(cpu_data);
    for (size_t i = 0; i < n; i++) {
      dst[i] = src[i] ? 1 : 0;
    }
  }

  // Clean up staging buffer
  wgpuBufferUnmap(staging);
  wgpuBufferDestroy(staging);
  wgpuBufferRelease(staging);

  // Store CPU pointer; GPU buffer remains valid for potential future GPU use
  wbuf.cpu_ptr = cpu_data;
  wbuf.cpu_dirty = false; // Just read from GPU, so CPU is in sync

  return cpu_data;
}

} // namespace allocator

size_t get_active_memory() {
  return wgpu::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return wgpu::allocator().get_peak_memory();
}
void reset_peak_memory() {
  wgpu::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return wgpu::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return wgpu::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return wgpu::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return wgpu::allocator().set_cache_limit(limit);
}
void clear_cache() {
  wgpu::allocator().clear_cache();
}

// Not supported in WebGPU.
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core
