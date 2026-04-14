// Copyright 2026 Apple Inc.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/backend/webgpu/worker.h"
#include "mlx/utils.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace mlx::core::wgpu {

// ---------------------------------------------------------------------------
// Phase 0 dispatch-stats counters (observability only, no behavior change).
//
// Global atomics, not CommandEncoder members, because:
//   1. CommandEncoder instances are per-stream and their WGPU encoder handle
//      is released on every commit() — the object survives, but counters that
//      intentionally outlive individual commits still need to live at a
//      single, process-wide location so the FFI reader does not have to walk
//      Device::encoders_ (which would also need locking).
//   2. The packed-bf16 / sdpa-fallback toggles in allocator.cpp already use
//      the same "global atomic" pattern; we keep Phase 0 instrumentation
//      consistent with that pattern.
//
// These counters are read by mlx_wgpu_get_dispatch_stats / reset by
// mlx_wgpu_reset_dispatch_stats (see crates/mlx-sys/src/mlx_stream.cpp).
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> g_total_dispatches{0};
static std::atomic<uint64_t> g_total_pass_ends{0};

uint64_t get_total_dispatches() {
  return g_total_dispatches.load(std::memory_order_relaxed);
}

uint64_t get_total_pass_ends() {
  return g_total_pass_ends.load(std::memory_order_relaxed);
}

void reset_dispatch_stats() {
  g_total_dispatches.store(0, std::memory_order_relaxed);
  g_total_pass_ends.store(0, std::memory_order_relaxed);
}

///////////////////////////////////////////////////////////////////////////////
// UniformBufferPool implementation
///////////////////////////////////////////////////////////////////////////////

WGPUBuffer UniformBufferPool::acquire(
    WGPUQueue queue,
    const void* data,
    size_t size) {
  size_t aligned = align_size(size);
  WGPUBuffer buf = nullptr;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_lists_.find(aligned);
    if (it != free_lists_.end() && !it->second.empty()) {
      buf = it->second.back();
      it->second.pop_back();
    }
  }

  if (!buf) {
    // Create a new buffer with Uniform | CopyDst usage
    WGPUBufferDescriptor desc = {};
    desc.size = aligned;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = false;
    buf = wgpuDeviceCreateBuffer(device().gpu_device(), &desc);
    if (!buf) {
      throw std::runtime_error(
          "[WebGPU] Failed to create uniform buffer for pool");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    buf_sizes_[buf] = aligned;
  }

  // Write data into the buffer
  wgpuQueueWriteBuffer(queue, buf, 0, data, size);
  return buf;
}

void UniformBufferPool::release(WGPUBuffer buf) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buf_sizes_.find(buf);
  if (it != buf_sizes_.end()) {
    free_lists_[it->second].push_back(buf);
  }
}

UniformBufferPool::~UniformBufferPool() {
  for (auto& [size, bufs] : free_lists_) {
    for (auto buf : bufs) {
      wgpuBufferDestroy(buf);
      wgpuBufferRelease(buf);
    }
  }
  free_lists_.clear();
  buf_sizes_.clear();
}

///////////////////////////////////////////////////////////////////////////////
// Device implementation
///////////////////////////////////////////////////////////////////////////////

Device::Device() {
  // Create instance
  WGPUInstanceDescriptor instance_desc = {};
  instance_ = wgpuCreateInstance(&instance_desc);
  if (!instance_) {
    return; // WebGPU not available
  }

  // Request adapter (blocking via polling)
  struct AdapterUserData {
    WGPUAdapter adapter = nullptr;
    bool done = false;
    std::mutex mtx;
    std::condition_variable cv;
  };
  AdapterUserData adapter_ud;

  WGPURequestAdapterOptions adapter_opts = {};
  adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;

  wgpuInstanceRequestAdapter(
      instance_,
      &adapter_opts,
      [](WGPURequestAdapterStatus status,
         WGPUAdapter adapter,
         char const* message,
         void* userdata) {
        auto* ud = static_cast<AdapterUserData*>(userdata);
        if (status == WGPURequestAdapterStatus_Success) {
          ud->adapter = adapter;
        } else {
          fprintf(
              stderr,
              "[WebGPU] Adapter request failed: %s\n",
              message ? message : "unknown");
        }
        {
          std::lock_guard<std::mutex> lock(ud->mtx);
          ud->done = true;
        }
        ud->cv.notify_one();
      },
      &adapter_ud);

  // Poll until adapter request completes
  {
    std::unique_lock<std::mutex> lock(adapter_ud.mtx);
    while (!adapter_ud.done) {
      lock.unlock();
      poll_instance(instance_);
      lock.lock();
    }
  }

  adapter_ = adapter_ud.adapter;
  if (!adapter_) {
    wgpuInstanceRelease(instance_);
    instance_ = nullptr;
    return;
  }

  // Request device with desired limits (blocking via polling)
  struct DeviceUserData {
    WGPUDevice device = nullptr;
    bool done = false;
    std::mutex mtx;
    std::condition_variable cv;
  };
  DeviceUserData device_ud;

  // Set required limits - start with all zeros and set only what we need
  WGPURequiredLimits required_limits = {};
  required_limits.limits.maxStorageBuffersPerShaderStage = 16;
  required_limits.limits.maxComputeWorkgroupSizeX = 256;
  required_limits.limits.maxComputeWorkgroupSizeY = 256;
  required_limits.limits.maxComputeWorkgroupSizeZ = 64;
  required_limits.limits.maxComputeInvocationsPerWorkgroup = 256;
  required_limits.limits.maxComputeWorkgroupsPerDimension = 65535;
  required_limits.limits.maxBindGroups = 4;
  required_limits.limits.maxBindingsPerBindGroup = 16;

  // Query adapter limits (bridge stub handles this for WASM builds)
  WGPUSupportedLimits adapter_limits = {};
  wgpuAdapterGetLimits(adapter_, &adapter_limits);
  auto clamp = [](uint64_t desired, uint64_t supported) -> uint64_t {
    return desired < supported ? desired : supported;
  };
  required_limits.limits.maxBufferSize =
      clamp(1ULL << 30, adapter_limits.limits.maxBufferSize);
  required_limits.limits.maxStorageBufferBindingSize =
      clamp(1ULL << 30, adapter_limits.limits.maxStorageBufferBindingSize);

  // Check optional features (bridge stub resolves locally for WASM)
  bool has_f16 = wgpuAdapterHasFeature(adapter_, WGPUFeatureName_ShaderF16);
  bool has_sg = wgpuAdapterHasFeature(adapter_, WGPUFeatureName_Subgroups);

  WGPUFeatureName optional_features[2] = {};
  size_t feature_count = 0;
  if (has_f16) optional_features[feature_count++] = WGPUFeatureName_ShaderF16;
  if (has_sg) optional_features[feature_count++] = WGPUFeatureName_Subgroups;

  WGPUDeviceDescriptor device_desc = {};
  device_desc.requiredLimits = &required_limits;
  device_desc.defaultQueue.label = "MLX Default Queue";
  device_desc.requiredFeatureCount = feature_count;
  device_desc.requiredFeatures = optional_features;

  wgpuAdapterRequestDevice(
      adapter_,
      &device_desc,
      [](WGPURequestDeviceStatus status,
         WGPUDevice device,
         char const* message,
         void* userdata) {
        auto* ud = static_cast<DeviceUserData*>(userdata);
        if (status == WGPURequestDeviceStatus_Success) {
          ud->device = device;
        } else {
          fprintf(
              stderr,
              "[WebGPU] Device request failed: %s\n",
              message ? message : "unknown");
        }
        {
          std::lock_guard<std::mutex> lock(ud->mtx);
          ud->done = true;
        }
        ud->cv.notify_one();
      },
      &device_ud);

  // Poll until device request completes
  {
    std::unique_lock<std::mutex> lock(device_ud.mtx);
    while (!device_ud.done) {
      lock.unlock();
      poll_instance(instance_);
      lock.lock();
    }
  }

  device_ = device_ud.device;
  if (!device_) {
    wgpuAdapterRelease(adapter_);
    adapter_ = nullptr;
    wgpuInstanceRelease(instance_);
    instance_ = nullptr;
    return;
  }

  has_shader_f16_ = has_f16;
  has_subgroups_ = has_sg;

  // Set error callback on device
  wgpuDeviceSetUncapturedErrorCallback(
      device_,
      [](WGPUErrorType type, char const* message, void* /* userdata */) {
        if (type != WGPUErrorType_NoError) {
          fprintf(
              stderr,
              "[WebGPU] Uncaptured error (%d): %s\n",
              static_cast<int>(type),
              message ? message : "");
        }
      },
      nullptr);

  // Set device lost callback
  wgpuDeviceSetDeviceLostCallback(
      device_,
      [](WGPUDeviceLostReason reason,
         char const* message,
         void* /* userdata */) {
        if (reason != WGPUDeviceLostReason_Destroyed) {
          fprintf(
              stderr,
              "[WebGPU] Device lost: %s\n",
              message ? message : "");
        }
      },
      nullptr);

  queue_ = wgpuDeviceGetQueue(device_);
}

Device::~Device() {
  // Clear encoders first (they hold worker threads)
  encoders_.clear();

  // Release shader cache
  for (auto& [key, mod] : shader_cache_) {
    if (mod) {
      wgpuShaderModuleRelease(mod);
    }
  }
  shader_cache_.clear();

  // Release pipeline cache
  for (auto& [key, entry] : pipeline_cache_) {
    if (entry.layout) {
      wgpuBindGroupLayoutRelease(entry.layout);
    }
    if (entry.pipeline) {
      wgpuComputePipelineRelease(entry.pipeline);
    }
  }
  pipeline_cache_.clear();

  if (queue_) {
    wgpuQueueRelease(queue_);
  }
  if (device_) {
    wgpuDeviceRelease(device_);
  }
  if (adapter_) {
    wgpuAdapterRelease(adapter_);
  }
  if (instance_) {
    wgpuInstanceRelease(instance_);
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  std::lock_guard<std::mutex> lock(encoder_mutex_);
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    it = encoders_.emplace(s.index, std::make_unique<CommandEncoder>(*this))
             .first;
  }
  return *it->second;
}

Device::PipelineEntry Device::get_or_create_pipeline(
    const std::string& key,
    WGPUShaderModule shader_module,
    const char* entry_point) {
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  auto it = pipeline_cache_.find(key);
  if (it != pipeline_cache_.end()) {
    return it->second;
  }

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.compute.module = shader_module;
  pipeline_desc.compute.entryPoint = entry_point;
  pipeline_desc.layout = nullptr; // auto layout

  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device_, &pipeline_desc);
  if (!pipeline) {
    throw std::runtime_error(
        "[WebGPU] Failed to create compute pipeline: " + key);
  }

  WGPUBindGroupLayout layout =
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
  if (!layout) {
    throw std::runtime_error(
        "[WebGPU] Failed to get bind group layout: " + key);
  }

  PipelineEntry entry{pipeline, layout};
  pipeline_cache_[key] = entry;
  return entry;
}

WGPUShaderModule Device::get_or_create_shader_module(
    const std::string& key,
    const std::function<std::string()>& source_builder) {
  std::lock_guard<std::mutex> lock(shader_mutex_);
  auto it = shader_cache_.find(key);
  if (it != shader_cache_.end()) {
    return it->second;
  }

  // Cache miss: generate the source via the builder lambda
  std::string source = source_builder();

  WGPUShaderModuleWGSLDescriptor wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgsl_desc.code = source.c_str();

  WGPUShaderModuleDescriptor desc = {};
  desc.nextInChain = &wgsl_desc.chain;
  desc.label = key.c_str();

  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device_, &desc);
  if (!mod) {
    throw std::runtime_error(
        "[WebGPU] Failed to create shader module: " + key);
  }

  shader_cache_[key] = mod;
  return mod;
}

Device& device() {
  // Heap-allocated singleton, intentionally leaked (same as CUDA pattern).
  static auto* dev = new Device();
  return *dev;
}

CommandEncoder& get_command_encoder(Stream s) {
  return device().get_command_encoder(s);
}

void Device::flush_all_encoders() {
  std::lock_guard<std::mutex> lock(encoder_mutex_);
  for (auto& [_, enc] : encoders_) {
    enc->commit();
  }
}

void flush_all_encoders() {
  device().flush_all_encoders();
}

// Upload CPU data to GPU with format conversion for types where
// GPU element size differs from CPU (bf16→f32, bool→u32).
//
// For bf16 we always upload the WHOLE owning CPU buffer, not just the
// `data_size` elements of the (possibly sliced) view that triggered the
// upload. If a view of the second half of a bf16 tensor is the first
// caller into wgpu_buffer(), uploading only `data_size` elements would
// leave the actual region the view references unwritten (zeroed), and
// kernel reads at the view's offset would then read garbage. Using the
// buffer's `cpu_bytes` guarantees the full owning buffer is uploaded once.
static void upload_with_conversion(
    WebGPUBuffer* buf,
    Dtype::Val dtype_val,
    size_t data_size) {
  auto queue = device().gpu_queue();

  if (dtype_val == Dtype::Val::bfloat16) {
    // Owning CPU allocation holds `cpu_bytes` bytes of bf16 data
    // (2 bytes per element). Use this count — not the view's data_size —
    // so slice-triggered uploads still cover the entire buffer.
    size_t full_n =
        buf->cpu_bytes > 0 ? (buf->cpu_bytes / 2) : data_size;
    if (buf->storage_mode == StorageMode::PackedBf16) {
      // Pack bf16 pairs into u32 (lo | hi << 16). Odd element counts pad
      // the trailing slot with a zero — the kernel masks out-of-range loads.
      size_t n_u32 = (full_n + 1) / 2;
      size_t byte_size = n_u32 * 4;
      // Cap to the actual GPU buffer capacity in case rounding pushed it
      // above wbuf->size (shouldn't happen, but be defensive).
      if (byte_size > buf->size) {
        byte_size = buf->size;
        n_u32 = byte_size / 4;
      }
      std::vector<uint32_t> tmp(n_u32, 0u);
      auto* src = static_cast<const uint8_t*>(buf->cpu_ptr);
      // The source is a contiguous block of bf16 values. bf16 elements are
      // 2 bytes each; we copy two at a time into the low / high halves of
      // each u32. A byte-by-byte path handles non-4B-aligned inputs.
      size_t n_pack = n_u32 * 2;
      if (n_pack > full_n) n_pack = full_n;
      for (size_t i = 0; i < n_pack; ++i) {
        uint16_t v = static_cast<uint16_t>(src[2 * i]) |
            (static_cast<uint16_t>(src[2 * i + 1]) << 8);
        uint32_t& slot = tmp[i >> 1];
        if ((i & 1u) == 0u) {
          slot = (slot & 0xFFFF0000u) | static_cast<uint32_t>(v);
        } else {
          slot = (slot & 0x0000FFFFu) |
              (static_cast<uint32_t>(v) << 16);
        }
      }
      wgpuQueueWriteBuffer(queue, buf->buffer, 0, tmp.data(), byte_size);
      return;
    }
    // Expand bf16 (2 bytes) → f32 (4 bytes)
    size_t n_up = full_n;
    if (n_up * 4 > buf->size) {
      n_up = buf->size / 4;
    }
    std::vector<float> tmp(n_up);
    auto* src = static_cast<const bfloat16_t*>(buf->cpu_ptr);
    for (size_t i = 0; i < n_up; i++) {
      tmp[i] = static_cast<float>(src[i]);
    }
    wgpuQueueWriteBuffer(queue, buf->buffer, 0, tmp.data(), n_up * 4);
  } else if (dtype_val == Dtype::Val::bool_) {
    // Expand bool (1 byte) → u32 (4 bytes)
    std::vector<uint32_t> tmp(data_size);
    auto* src = static_cast<const uint8_t*>(buf->cpu_ptr);
    for (size_t i = 0; i < data_size; i++) {
      tmp[i] = src[i] ? 1u : 0u;
    }
    wgpuQueueWriteBuffer(queue, buf->buffer, 0, tmp.data(), data_size * 4);
  } else {
    // Direct copy for types where CPU and GPU sizes match
    wgpuQueueWriteBuffer(queue, buf->buffer, 0, buf->cpu_ptr, buf->size);
  }
}

WGPUBuffer wgpu_buffer(const array& arr) {
  auto* buf = const_cast<WebGPUBuffer*>(
      static_cast<const WebGPUBuffer*>(arr.buffer().ptr()));
  if (!buf || !buf->buffer) {
    throw std::runtime_error("[WebGPU] wgpu_buffer: null buffer");
  }

  // Track dtype for download conversion in raw_ptr()
  buf->dtype_val = arr.dtype().val();

  // NOTE: there is deliberately no auto-opt-in to StorageMode::PackedBf16
  // here. The packed path is only reachable through explicit "weight upload"
  // entry points that set the storage mode at the buffer's birth:
  //   * mlx_array_from_bfloat16 — CPU bytes → WebGPUBuffer (test harness)
  //   * mlx_array_from_gpu_buffer(..., packed_bf16=true) — JS-side packed
  //     GPU handle import (demo weight-upload path in gpu-worker.ts)
  // Flipping intermediate/compute bf16 tensors would break readback because
  // PackedBf16 is upload-only (raw_ptr() throws on packed buffers).

  // Ensure GPU buffer is large enough for GPU-side element sizes.
  // Input arrays allocated for CPU types (bf16=2B, bool=1B) may be too small
  // for GPU types (f32=4B, u32=4B). Recreate the buffer if needed.
  //
  // For compute-output buffers (the hot path during decode), the caller
  // already passes a GPU-sized byte count to allocator::malloc — i.e.
  // `wgpu_alloc_size(out) = nelements * wgpu_itemsize`. The GPU buffer is
  // already the right size; `arr.data_size() * wgpu_itemsize()` gives the
  // exact `needed` and never triggers a recreate. This is the baseline path
  // and MUST stay cheap (called per matmul per layer per token).
  //
  // The slice-safety case (described below) only matters when the buffer
  // was populated from raw CPU bytes via from_bfloat16_bytes / from_bool —
  // in that case `cpu_dirty && cpu_ptr` is true and `cpu_bytes` holds the
  // raw bf16/bool byte count of the owning allocation, which we use to size
  // the full owning buffer so subsequent slice uploads cover the whole
  // region.
  //
  // Slice-safety rationale: if the first wgpu_buffer() call for an owning
  // bf16/bool allocation comes from a *slice view*, arr.data_size() reflects
  // only the view's span. Using that for `needed` would make the GPU buffer
  // too small for the bf16→f32 (or bool→u32) expansion of the OWNING
  // allocation, and upload_with_conversion() would silently truncate the
  // upload — leaving the rest of the buffer uninitialised.
  size_t needed;
  if (buf->storage_mode == StorageMode::PackedBf16) {
    // Packed bf16: 2 bf16 per u32. Source has cpu_bytes/2 bf16 elements
    // → ((cpu_bytes/2 + 1) / 2) * 4 bytes of u32 storage.
    if (buf->cpu_bytes > 0) {
      size_t full_n = buf->cpu_bytes / 2;
      needed = ((full_n + 1) / 2) * 4;
    } else {
      needed = ((arr.size() + 1) / 2) * 4;
    }
  } else if (
      arr.dtype().val() == Dtype::Val::bfloat16 && buf->cpu_bytes > 0 &&
      buf->cpu_dirty && buf->cpu_ptr) {
    // Raw bf16 bytes pending upload from CPU: size for the full owning
    // allocation. cpu_bytes here is the literal bf16 byte count (2B/elem).
    needed = (buf->cpu_bytes / 2) * 4;
  } else if (
      arr.dtype().val() == Dtype::Val::bool_ && buf->cpu_bytes > 0 &&
      buf->cpu_dirty && buf->cpu_ptr) {
    // Raw bool bytes pending upload from CPU: size for the full owning
    // allocation. cpu_bytes here is the literal bool byte count (1B/elem).
    needed = buf->cpu_bytes * 4;
  } else {
    needed = arr.data_size() * wgpu_itemsize(arr.dtype());
  }
  if (needed > buf->size) {
    if (buf->buffer) {
      wgpuBufferDestroy(buf->buffer);
      wgpuBufferRelease(buf->buffer);
    }
    // Round up the same way as WebGPUAllocator::malloc
    size_t new_size = needed;
    constexpr size_t page = 16384;
    if (new_size <= 8) {
      new_size = 8;
    } else if (new_size < page) {
      // next power of 2
      new_size--;
      new_size |= new_size >> 1;
      new_size |= new_size >> 2;
      new_size |= new_size >> 4;
      new_size |= new_size >> 8;
      new_size |= new_size >> 16;
      new_size++;
    } else {
      new_size = page * ((new_size + page - 1) / page);
    }
    WGPUBufferDescriptor desc = {};
    desc.size = new_size;
    desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
        WGPUBufferUsage_CopySrc;
    desc.mappedAtCreation = false;
    buf->buffer = wgpuDeviceCreateBuffer(device().gpu_device(), &desc);
    if (!buf->buffer) {
      throw std::runtime_error("[WebGPU] Failed to resize buffer for GPU type");
    }
    buf->size = new_size;
    buf->gpu_has_data = false;
  }

  // If CPU data was written (e.g. by array::init) but never uploaded to GPU,
  // upload it now with format conversion for wider GPU types.
  if (buf->cpu_dirty && buf->cpu_ptr && buf->buffer) {
    upload_with_conversion(buf, arr.dtype().val(), arr.data_size());
    buf->cpu_dirty = false;
    buf->gpu_has_data = true;
    // Free the CPU copy after upload — it's now redundant, and keeping it
    // risks returning stale data from raw_ptr() if the buffer is later
    // used as a GPU compute output.
    std::free(buf->cpu_ptr);
    buf->cpu_ptr = nullptr;
  }
  // Safety: if cpu_ptr exists with data but gpu_has_data is false,
  // ensure the data gets uploaded. This handles cases where
  // eval() wrote to CPU memory (e.g. NumberOfElements) but the
  // dirty flag was somehow missed.
  if (!buf->gpu_has_data && buf->cpu_ptr && !buf->cpu_dirty) {
    upload_with_conversion(buf, arr.dtype().val(), arr.data_size());
    buf->gpu_has_data = true;
    std::free(buf->cpu_ptr);
    buf->cpu_ptr = nullptr;
  }
  return buf->buffer;
}

///////////////////////////////////////////////////////////////////////////////
// CommandEncoder implementation
///////////////////////////////////////////////////////////////////////////////

CommandEncoder::CommandEncoder(Device& d)
    : device_(d), worker_(std::make_unique<Worker>()) {
  max_ops_per_commit_ = env::max_ops_per_buffer(512);
  max_mb_per_commit_ = env::max_mb_per_buffer(512);
}

CommandEncoder::~CommandEncoder() {
  end_compute_pass();
  if (encoder_) {
    wgpuCommandEncoderRelease(encoder_);
    encoder_ = nullptr;
  }
}

void CommandEncoder::set_input_array(const array& arr) {
  bytes_tracked_ += arr.data_size() * arr.itemsize();
}

void CommandEncoder::set_output_array(const array& arr) {
  bytes_tracked_ += arr.data_size() * arr.itemsize();
  // Mark that this buffer will have meaningful GPU data after compute.
  // Invalidate any cached CPU pointer — the GPU is about to overwrite
  // the buffer contents, so the old CPU copy would be stale.
  auto* buf = const_cast<WebGPUBuffer*>(
      static_cast<const WebGPUBuffer*>(arr.buffer().ptr()));
  if (buf) {
    buf->gpu_has_data = true;
    if (buf->cpu_ptr) {
      std::free(buf->cpu_ptr);
      buf->cpu_ptr = nullptr;
    }
    buf->cpu_dirty = false;
  }
}

void CommandEncoder::ensure_active() {
  if (!encoder_) {
    WGPUCommandEncoderDescriptor enc_desc = {};
    encoder_ = wgpuDeviceCreateCommandEncoder(device_.gpu_device(), &enc_desc);
    if (!encoder_) {
      throw std::runtime_error("[WebGPU] Failed to create command encoder");
    }
  }
  if (!compute_pass_) {
    WGPUComputePassDescriptor pass_desc = {};
    compute_pass_ = wgpuCommandEncoderBeginComputePass(encoder_, &pass_desc);
    if (!compute_pass_) {
      throw std::runtime_error("[WebGPU] Failed to create compute pass");
    }
  }
}

void CommandEncoder::end_compute_pass() {
  if (compute_pass_) {
    // Phase 0 instrumentation: count actual pass-end events (i.e., only
    // when we had an open pass). See g_total_pass_ends above.
    g_total_pass_ends.fetch_add(1, std::memory_order_relaxed);
    wgpuComputePassEncoderEnd(compute_pass_);
    wgpuComputePassEncoderRelease(compute_pass_);
    compute_pass_ = nullptr;
    // Reset cached state — new compute pass needs fresh setPipeline/setBindGroup
    last_pipeline_ = nullptr;
    last_bind_group_ = nullptr;
  }
}

void CommandEncoder::dispatch_compute(
    WGPUComputePipeline pipeline,
    const std::vector<WGPUBindGroup>& bind_groups,
    uint32_t x,
    uint32_t y,
    uint32_t z) {
  ensure_active();

  // Skip redundant setPipeline if same as last dispatch
  if (pipeline != last_pipeline_) {
    wgpuComputePassEncoderSetPipeline(compute_pass_, pipeline);
    last_pipeline_ = pipeline;
  }
  for (uint32_t i = 0; i < bind_groups.size(); ++i) {
    wgpuComputePassEncoderSetBindGroup(
        compute_pass_, i, bind_groups[i], 0, nullptr);
  }
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass_, x, y, z);
  op_count_++;
  // Phase 0 instrumentation: cumulative dispatches across all encoders/commits.
  g_total_dispatches.fetch_add(1, std::memory_order_relaxed);
  // End compute pass after each dispatch to avoid WebGPU buffer aliasing
  // violations. WebGPU disallows a buffer being both read-only and read-write
  // storage within a single compute pass. Ending the pass after each dispatch
  // ensures each operation gets its own synchronization scope.
  end_compute_pass();
}

void CommandEncoder::dispatch_compute(
    WGPUComputePipeline pipeline,
    WGPUBindGroup bind_group,
    uint32_t x,
    uint32_t y,
    uint32_t z) {
  ensure_active();

  wgpuComputePassEncoderSetPipeline(compute_pass_, pipeline);
  wgpuComputePassEncoderSetBindGroup(
      compute_pass_, 0, bind_group, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass_, x, y, z);
  op_count_++;
  // Phase 0 instrumentation: cumulative dispatches across all encoders/commits.
  g_total_dispatches.fetch_add(1, std::memory_order_relaxed);
  // End compute pass after each dispatch (same reason as above)
  end_compute_pass();
}

void CommandEncoder::copy_buffer_to_buffer(
    WGPUBuffer src,
    uint64_t src_offset,
    WGPUBuffer dst,
    uint64_t dst_offset,
    uint64_t size) {
  if (size == 0) {
    return;
  }
  // Make sure we have a raw command encoder but NOT an open compute pass —
  // wgpuCommandEncoderCopyBufferToBuffer is only valid outside a pass.
  ensure_active();
  end_compute_pass();
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder_, src, src_offset, dst, dst_offset, size);
  // Count as an op so needs_commit() still flushes at the right time, but
  // deliberately NOT as a shader dispatch — Phase 0's g_total_dispatches
  // tracks kernel launches only. The whole point of this fast path is that
  // no shader ran.
  op_count_++;
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

bool CommandEncoder::needs_commit() {
  return (op_count_ > max_ops_per_commit_) ||
      ((bytes_tracked_ >> 20) > static_cast<size_t>(max_mb_per_commit_));
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }

  if (op_count_ > 0 && encoder_) {
    // End compute pass
    end_compute_pass();

    // Finish and submit
    WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(encoder_, nullptr);
    if (!cmd_buf) {
      wgpuCommandEncoderRelease(encoder_);
      encoder_ = nullptr;
      throw std::runtime_error("[WebGPU] Failed to finish command encoder");
    }
    wgpuQueueSubmit(device_.gpu_queue(), 1, &cmd_buf);
    wgpuCommandBufferRelease(cmd_buf);

    wgpuCommandEncoderRelease(encoder_);
    encoder_ = nullptr;
  }

  // Put completion handlers in a batch.
  worker_->commit(device_.gpu_queue());

  op_count_ = 0;
  bytes_tracked_ = 0;
}

void CommandEncoder::synchronize() {
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  f.wait();
}

} // namespace mlx::core::wgpu
