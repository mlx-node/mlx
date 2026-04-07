// Copyright 2026 Apple Inc.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/backend/webgpu/worker.h"
#include "mlx/utils.h"

#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>

#if defined(__wasm__)
extern "C" void mlx_wgpu_fused_dispatch(
    WGPUComputePassEncoder, WGPUComputePipeline, WGPUBindGroup,
    uint32_t, uint32_t, uint32_t);
#endif

namespace mlx::core::wgpu {

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

#if defined(__wasm__)
  // WASI: adapter query APIs may not be available in the JS bridge.
  // Use conservative buffer limits (256 MB, the WebGPU spec minimum).
  required_limits.limits.maxBufferSize = 1ULL << 28;
  required_limits.limits.maxStorageBufferBindingSize = 1ULL << 28;
  bool has_f16 = false;
#else
  // Query adapter limits so we can clamp our requests to what's supported.
  WGPUSupportedLimits adapter_limits = {};
  wgpuAdapterGetLimits(adapter_, &adapter_limits);
  auto clamp = [](uint64_t desired, uint64_t supported) -> uint64_t {
    return desired < supported ? desired : supported;
  };
  required_limits.limits.maxBufferSize =
      clamp(1ULL << 30, adapter_limits.limits.maxBufferSize);
  required_limits.limits.maxStorageBufferBindingSize =
      clamp(1ULL << 30, adapter_limits.limits.maxStorageBufferBindingSize);

  // Check if adapter supports shader-f16 and request it if available.
  bool has_f16 = wgpuAdapterHasFeature(adapter_, WGPUFeatureName_ShaderF16);
#endif

  WGPUFeatureName optional_features[] = {WGPUFeatureName_ShaderF16};

  WGPUDeviceDescriptor device_desc = {};
  device_desc.requiredLimits = &required_limits;
  device_desc.defaultQueue.label = "MLX Default Queue";
  if (has_f16) {
    device_desc.requiredFeatureCount = 1;
    device_desc.requiredFeatures = optional_features;
  }

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
}

void CommandEncoder::dispatch_compute(
    WGPUComputePipeline pipeline,
    WGPUBindGroup bind_group,
    uint32_t x,
    uint32_t y,
    uint32_t z) {
  ensure_active();

#if defined(__wasm__)
  // Fused dispatch: single RPC roundtrip instead of 3 separate calls
  mlx_wgpu_fused_dispatch(compute_pass_, pipeline, bind_group, x, y, z);
  last_pipeline_ = pipeline;
  last_bind_group_ = bind_group;
#else
  // Skip redundant setPipeline if same as last dispatch
  if (pipeline != last_pipeline_) {
    wgpuComputePassEncoderSetPipeline(compute_pass_, pipeline);
    last_pipeline_ = pipeline;
  }
  // Skip redundant setBindGroup if same as last dispatch
  if (bind_group != last_bind_group_) {
    wgpuComputePassEncoderSetBindGroup(
        compute_pass_, 0, bind_group, 0, nullptr);
    last_bind_group_ = bind_group;
  }
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass_, x, y, z);
#endif
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
