// Copyright 2026 Apple Inc.

#include "mlx/backend/webgpu/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/dtype.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace wgpu {

// Runtime opt-in for packed-bf16 storage. Browser has no env vars, so this
// is plumbed from the TS init message via a FFI call.
static std::atomic<bool> g_packed_bf16_enabled{false};

void set_packed_bf16_enabled(bool enabled) {
  g_packed_bf16_enabled.store(enabled, std::memory_order_relaxed);
}

bool packed_bf16_enabled() {
  return g_packed_bf16_enabled.load(std::memory_order_relaxed);
}

// Runtime force-on for the SDPA decomposed fallback. When set, SDPA always
// reports use_fallback() == true so the tile / vector kernels never fire.
// Used by the demo app for TTFT A/B comparisons.
static std::atomic<bool> g_sdpa_fallback_forced{false};

void set_sdpa_fallback_forced(bool enabled) {
  g_sdpa_fallback_forced.store(enabled, std::memory_order_relaxed);
}

bool sdpa_fallback_forced() {
  return g_sdpa_fallback_forced.load(std::memory_order_relaxed);
}

// Flip the storage mode of an upload-pending bf16 array to PackedBf16 when
// eligible (bf16 dtype, size >= threshold, packed_bf16_enabled() is true,
// and the buffer is still in "upload-pending" state). Called from the Rust
// `fromBfloat16Bytes` constructor immediately after the array is created
// (via the FFI bridge in mlx_array_ops.cpp), so the first wgpu_buffer() call
// uploads via the packed path. A no-op if any precondition fails — safe to
// call unconditionally from a constructor.
//
// ALLOC-F006: the read of gpu_has_data / cpu_dirty / cpu_ptr AND the write
// to storage_mode both run under allocator::mutex_. This prevents a
// torn-flip race with another thread that is simultaneously mutating the
// same WebGPUBuffer (e.g. the allocator free path that recycles to cache).
// The happens-before with any GPU compute that later dispatches against
// this buffer is provided by the encoder's commit/submit barrier, which
// always follows the storage_mode write on the caller's timeline.
bool try_opt_in_packed_bf16(array& arr, size_t min_elements) {
  if (!packed_bf16_enabled()) {
    return false;
  }
  if (arr.dtype() != bfloat16) {
    return false;
  }
  if (arr.size() < min_elements) {
    return false;
  }
  auto* wbuf = static_cast<WebGPUBuffer*>(arr.buffer().ptr());
  if (!wbuf) {
    return false;
  }
  std::lock_guard<std::mutex> lock(allocator().mutex_);
  // Only flip on buffers that are still in "upload-pending" state — never
  // touch a buffer that has already been used as a GPU compute input/output.
  if (wbuf->gpu_has_data) {
    return false;
  }
  if (!wbuf->cpu_dirty && !wbuf->cpu_ptr) {
    return false;
  }
  wbuf->storage_mode = StorageMode::PackedBf16;
  return true;
}

// Unconditionally mark a buffer as PackedBf16 storage. Used by the JS-side
// weight upload path in gpu-worker.ts, which has already packed the bf16
// data into u32 slots by the time it reaches the import call. The caller is
// responsible for making sure the underlying WGPUBuffer contents actually
// match the packed layout; this only flips the bookkeeping flag so the
// dispatcher and readback guards behave correctly.
bool mark_buffer_packed_bf16(array& arr) {
  if (arr.dtype() != bfloat16) {
    return false;
  }
  auto* wbuf = static_cast<WebGPUBuffer*>(arr.buffer().ptr());
  if (!wbuf) {
    return false;
  }
  // ALLOC-F006: serialize with any concurrent flip / reset. See the rationale
  // comment on try_opt_in_packed_bf16 above.
  std::lock_guard<std::mutex> lock(allocator().mutex_);
  wbuf->storage_mode = StorageMode::PackedBf16;
  return true;
}

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
    // ALLOC-F010: zero-size sentinel. The caller gets a WebGPUBuffer whose
    // size==0 / buffer==nullptr signals "no GPU allocation was performed".
    // The `free` path recognises this by `buf->size == 0` and deletes the
    // shell without touching the (absent) WGPUBuffer. Using designated
    // initializers (rather than positional) so the intent is explicit and
    // adding a future field to WebGPUBuffer doesn't silently shift the
    // positional semantics here.
    return Buffer{new WebGPUBuffer{
        .buffer = nullptr,
        .size = 0,
        .cpu_ptr = nullptr}};
  }

  // Remember the caller's requested (unrounded) byte count. This is the
  // "logical CPU byte count" we stash on WebGPUBuffer::cpu_bytes so that
  // wgpu_buffer()'s bf16 sizing and upload_with_conversion()'s pack loop
  // use the exact number of bytes the caller intends to populate — not the
  // rounded-up allocation size, which overestimates for types whose GPU
  // representation is wider than the CPU representation (bf16→f32, bool→u32).
  const size_t requested_size = size;

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
    // Reset state for reused buffer. storage_mode MUST be reset — otherwise
    // a buffer previously used as PackedBf16 leaks its mode to unrelated
    // future allocations, corrupting subsequent uploads and readback logic.
    //
    // ALLOC-F002: cpu_ptr is also asserted null on cache insert, so in
    // practice the std::free here is a no-op — kept as a belt-and-braces
    // cleanup in case an insert path regresses and forgets to drop its CPU
    // shadow.
    if (buf->cpu_ptr) {
      std::free(buf->cpu_ptr);
      buf->cpu_ptr = nullptr;
    }
    // Set cpu_bytes to the NEW caller's requested size, not the cache
    // slot's physical size (which may be larger from cache slack). Without
    // this, a stale or oversized value would feed back into wgpu_buffer's
    // `needed` sizing (which uses cpu_bytes for bf16) and
    // upload_with_conversion's full_n (which packs cpu_bytes/2 elements),
    // causing uploads to overrun or truncate the buffer.
    buf->cpu_bytes = requested_size;
    buf->cpu_dirty = false;
    buf->gpu_has_data = false;
    buf->storage_mode = StorageMode::Upconverted;
    buf->dtype_val = Dtype::Val::float32;
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

    // ALLOC-F003: use designated initializers so the field order in
    // WebGPUBuffer can evolve without silently shifting the semantics of
    // this brace-init. The previous positional form set
    //   {buffer, size, cpu_ptr, cpu_dirty, gpu_has_data}
    // which broke as soon as cpu_bytes was inserted before cpu_dirty.
    buf = new WebGPUBuffer{
        .buffer = gpu_buf,
        .size = size,
        .cpu_ptr = nullptr,
        // Record the caller's intended CPU byte count. See comment above
        // in the cache-reuse branch for rationale. For bf16/bool, the
        // GPU-side element size (4B) is wider than the CPU-side size
        // (2B/1B) — storing the unrounded CPU request here keeps
        // wgpu_buffer()'s bf16 sizing and upload_with_conversion's pack
        // loop consistent with the caller's data.
        .cpu_bytes = requested_size,
        .cpu_dirty = false,
        .gpu_has_data = false};
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
    // ALLOC-F002: normalize bookkeeping fields at cache-insert time so
    // a cached buffer never carries stale per-allocation metadata. This
    // is symmetric with the reset-on-reuse block in malloc() above.
    // Telemetry or future eviction policy that inspects a cached buffer
    // then sees predictable defaults instead of the previous tenant's
    // dtype/mode.
    assert(buf->cpu_ptr == nullptr); // guaranteed by the branch above
    buf->cpu_bytes = 0;
    buf->cpu_dirty = false;
    buf->gpu_has_data = false;
    buf->dtype_val = Dtype::Val::float32;
    buf->storage_mode = StorageMode::Upconverted;
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

  // Packed-bf16 buffers are upload-only in Step A. Activation packing (with
  // an unpack-on-readback path) arrives in Step C; until then, attempting to
  // read a packed weight buffer back to CPU is a programming error.
  if (wbuf.storage_mode == wgpu::StorageMode::PackedBf16) {
    throw std::runtime_error(
        "[WebGPU] attempted to read packed weight buffer back to CPU; "
        "this buffer is not readable in Upconverted form");
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
    // NOTE: do NOT touch wbuf.cpu_bytes here. It was set by
    // WebGPUAllocator::malloc() to the caller's unrounded request size,
    // which is the correct CPU byte count. wbuf.size is the rounded-up
    // physical allocation size, and for bf16/bool would overestimate the
    // logical CPU byte count (e.g. bf16 stores 2B per element on CPU but
    // the GPU buffer is 4B per element, so setting cpu_bytes = wbuf.size
    // would double the count). Leave cpu_bytes alone.
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
  // ALLOC-F008: after the forward-safe in-place contraction, realloc the CPU
  // shadow down to the logical byte count. Holding onto the full GPU-sized
  // allocation (2x for bf16, 4x for bool) wastes heap capacity for the
  // lifetime of the array, which on WASM is scarce.
  if (wbuf.dtype_val == Dtype::Val::bfloat16) {
    // Contract f32 (4 bytes) -> bf16 (2 bytes) in-place (forward safe: 4B->2B)
    size_t n = wbuf.size / 4;
    auto* src = static_cast<const float*>(cpu_data);
    auto* dst = static_cast<bfloat16_t*>(cpu_data);
    for (size_t i = 0; i < n; i++) {
      dst[i] = static_cast<bfloat16_t>(src[i]);
    }
    size_t logical = n * sizeof(bfloat16_t);
    if (logical > 0 && logical < wbuf.size) {
      void* shrunk = std::realloc(cpu_data, logical);
      if (shrunk) cpu_data = shrunk;
      // realloc failure: keep the oversized block. It's still correct.
    }
  } else if (wbuf.dtype_val == Dtype::Val::bool_) {
    // Contract u32 (4 bytes) -> bool (1 byte) in-place (forward safe: 4B->1B)
    size_t n = wbuf.size / 4;
    auto* src = static_cast<const uint32_t*>(cpu_data);
    auto* dst = static_cast<uint8_t*>(cpu_data);
    for (size_t i = 0; i < n; i++) {
      dst[i] = src[i] ? 1 : 0;
    }
    size_t logical = n; // 1 byte / element
    if (logical > 0 && logical < wbuf.size) {
      void* shrunk = std::realloc(cpu_data, logical);
      if (shrunk) cpu_data = shrunk;
    }
  }

  // Clean up staging buffer
  wgpuBufferUnmap(staging);
  wgpuBufferDestroy(staging);
  wgpuBufferRelease(staging);

  // Store CPU pointer; GPU buffer remains valid for potential future GPU use.
  // NOTE: do NOT overwrite wbuf.cpu_bytes here. For buffers that came through
  // WebGPUAllocator::malloc(), cpu_bytes already holds the caller's original
  // unrounded request size (the logical CPU byte count), which is preserved
  // across GPU upload + readback round-trips. For bf16/bool buffers, setting
  // cpu_bytes = wbuf.size would overestimate the logical CPU byte count by
  // 2x/4x (wbuf.size is the wider GPU byte count). For buffers that never
  // went through malloc() (e.g. imported via mlx_array_from_gpu_buffer),
  // cpu_bytes stays 0, which makes wgpu_buffer()'s bf16 sizing fall back to
  // the safe `data_size * wgpu_itemsize` path.
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
