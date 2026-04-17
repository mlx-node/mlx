// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::wgpu {

class Device;
class Worker;

class CommandEncoder {
 public:
  explicit CommandEncoder(Device& d);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  // Dispatch a compute shader (multiple bind groups)
  void dispatch_compute(
      WGPUComputePipeline pipeline,
      const std::vector<WGPUBindGroup>& bind_groups,
      uint32_t x,
      uint32_t y = 1,
      uint32_t z = 1);

  // Dispatch a compute shader (single bind group, avoids vector allocation)
  void dispatch_compute(
      WGPUComputePipeline pipeline,
      WGPUBindGroup bind_group,
      uint32_t x,
      uint32_t y = 1,
      uint32_t z = 1);

  // Native buffer-to-buffer copy (no shader dispatch). Used by the metadata
  // fast path in copy.cpp for same-dtype linear copies that previously went
  // through the copy_v WGSL kernel. Must end any active compute pass because
  // wgpuCommandEncoderCopyBufferToBuffer is not valid inside a compute pass.
  void copy_buffer_to_buffer(
      WGPUBuffer src,
      uint64_t src_offset,
      WGPUBuffer dst,
      uint64_t dst_offset,
      uint64_t size);

  // Keep buffer alive until GPU work completes
  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  // Callback after GPU work finishes
  void add_completed_handler(std::function<void()> task);

  // Batch control
  bool needs_commit();
  void commit();

  // Commit + block until complete
  void synchronize();

  Device& device() {
    return device_;
  }

 private:
  // Ensure we have an active command encoder + compute pass
  void ensure_active();
  // End the current compute pass (if any)
  void end_compute_pass();

  Device& device_;
  std::unique_ptr<Worker> worker_;

  WGPUCommandEncoder encoder_{nullptr};
  WGPUComputePassEncoder compute_pass_{nullptr};
  int op_count_{0};
  int max_ops_per_commit_;
  int max_mb_per_commit_;
  size_t bytes_tracked_{0};
  std::vector<std::shared_ptr<array::Data>> temporaries_;

  // Track last-set pipeline/bindgroups to skip redundant RPC calls
  WGPUComputePipeline last_pipeline_{nullptr};
  WGPUBindGroup last_bind_group_{nullptr};
};

class UniformBufferPool {
 public:
  // Get a buffer of at least `size` bytes, write `data` into it
  WGPUBuffer acquire(WGPUQueue queue, const void* data, size_t size);
  // Return buffer to pool for reuse
  void release(WGPUBuffer buf);
  // Clean up all buffers
  ~UniformBufferPool();

 private:
  static constexpr size_t ALIGNMENT = 256;
  size_t align_size(size_t s) const {
    return (s + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
  }
  // Map from aligned_size -> free list of buffers
  std::unordered_map<size_t, std::vector<WGPUBuffer>> free_lists_;
  // Track allocated size for each buffer (for reuse lookup)
  std::unordered_map<WGPUBuffer, size_t> buf_sizes_;
  std::mutex mutex_;
};

class Device {
 public:
  Device();
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  bool is_valid() const {
    return device_ != nullptr;
  }

  WGPUDevice gpu_device() const {
    return device_;
  }
  WGPUQueue gpu_queue() const {
    return queue_;
  }
  WGPUInstance gpu_instance() const {
    return instance_;
  }
  WGPUAdapter gpu_adapter() const {
    return adapter_;
  }
  bool has_shader_f16() const { return has_shader_f16_; }
  bool has_subgroups() const { return has_subgroups_; }

  // Detected subgroup size from a WGSL probe kernel run at device creation.
  // Both fields are 0 when the device has no Subgroups feature OR when the
  // probe failed. The probe executes the kernel with workgroup_size(1), so it
  // samples a single invocation's `subgroupSize` builtin and stores that value
  // in both min and max; a device that varies subgroup size across invocations
  // will still report a single sample here (callers should disable the
  // subgroup code path via effective_subgroup_size() if tighter guarantees are
  // needed).
  uint32_t subgroup_size_min() const { return subgroup_size_min_; }
  uint32_t subgroup_size_max() const { return subgroup_size_max_; }

  CommandEncoder& get_command_encoder(Stream s);

  // Commit all pending command encoders (flush GPU work before readback)
  void flush_all_encoders();

  UniformBufferPool& uniform_pool() {
    return uniform_pool_;
  }

  struct PipelineEntry {
    WGPUComputePipeline pipeline;
    WGPUBindGroupLayout layout;
  };

  // Pipeline cache
  PipelineEntry get_or_create_pipeline(
      const std::string& key,
      WGPUShaderModule shader_module,
      const char* entry_point);

  // Shader module cache with lazy source generation.
  // The source_builder lambda is only called on cache miss.
  WGPUShaderModule get_or_create_shader_module(
      const std::string& key,
      const std::function<std::string()>& source_builder);

 private:
  // Dispatch a tiny WGSL compute kernel that reads `subgroupSize` and
  // stores the result into subgroup_size_min_/_max_. No-op (leaves both
  // fields at 0) if has_subgroups_ is false or any probe step fails.
  void probe_subgroup_size();

  WGPUInstance instance_{nullptr};
  WGPUAdapter adapter_{nullptr};
  WGPUDevice device_{nullptr};
  WGPUQueue queue_{nullptr};
  bool has_shader_f16_{false};
  bool has_subgroups_{false};
  uint32_t subgroup_size_min_{0};
  uint32_t subgroup_size_max_{0};

  std::unordered_map<int, std::unique_ptr<CommandEncoder>> encoders_;
  std::mutex encoder_mutex_;

  std::unordered_map<std::string, PipelineEntry> pipeline_cache_;
  std::mutex pipeline_mutex_;

  std::unordered_map<std::string, WGPUShaderModule> shader_cache_;
  std::mutex shader_mutex_;

  UniformBufferPool uniform_pool_;
};

// Get the singleton device
Device& device();

// Get the command encoder for a stream
CommandEncoder& get_command_encoder(Stream s);

// Flush all pending command encoders
void flush_all_encoders();

// Phase 0 dispatch-stats (observability only). These counters are global
// atomics incremented inside CommandEncoder::dispatch_compute /
// end_compute_pass; they are read + reset from the FFI bridge in mlx-sys
// so the browser can expose `?profile=1` metrics to the demo page.
uint64_t get_total_dispatches();
uint64_t get_total_pass_ends();
void reset_dispatch_stats();

} // namespace mlx::core::wgpu
