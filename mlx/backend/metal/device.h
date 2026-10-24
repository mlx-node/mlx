// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/backend/common/metal_kernel.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/device.h"

namespace mlx::core::metal {

using MTLFCList =
    std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

class Device;

class MLX_API CommandEncoder {
 public:
  CommandEncoder(Device& d, int index, ResidencySets& residency_sets);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc) : enc(enc) {
      enc.concurrent_ = true;
    }
    ~ConcurrentContext() {
      enc.concurrent_ = false;
      for (auto r : enc.concurrent_outputs_) {
        enc.prev_outputs_[r] = enc.barrier_epoch_;
      }
      enc.concurrent_outputs_.clear();
    }

   private:
    CommandEncoder& enc;
  };

  void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);
  void set_input_array(const array& a, int idx, int64_t offset = 0);
  // Scheduling estimate only, in data_size() storage units (not bytes).
  // Resource binding, hazards, residency and retention remain unchanged.
  // A repeated binding of this allocation restores its full original charge.
  void set_input_array_with_budget(
      const array& a, int idx, size_t touched_units, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void register_output_array(const array& a);

  struct WorkspaceSpec {
    Shape shape;
    Dtype dtype;
  };
  // One bounded profile per encoder; returns nullptr for an incompatible
  // request or inside ConcurrentContext. The borrowed vector and its array
  // descriptors/Data are immutable: only ordinary tracked GPU writes may
  // change contents. Do not donate/rebind storage or expose it as graph output.
  // The reference is only for the current eval. Storage survives command
  // completion/commit and is released after encoder teardown synchronizes.
  // No add_temporary: normal cross-encoder read/write fences must retain it.
  const std::vector<array>* try_workspace(
      uint64_t identity, const std::vector<WorkspaceSpec>& specs);

  void add_temporary(array arr);
  void add_temporaries(std::vector<array> arrays);

  // Retain `arr`'s input/sibling storage until the current command buffer
  // completes (skipping storage donated to `arr` itself). Accumulates across
  // evals so a single completed-handler per commit covers every retained
  // buffer (previously each eval built a set and attached its own ObjC
  // block — one allocation per primitive dispatch).
  void retain_inputs(const array& arr);

  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  // Register the argument buffer as an input before this dispatch.
  void dispatch_threadgroups_indirect(
      const MTL::Buffer* arguments,
      size_t offset,
      MTL::Size group_dims) {
    maybeInsertBarrier();
    ++buffer_ops_;
    get_command_encoder()->dispatchThreadgroups(arguments, offset, group_dims);
  }
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void maybeInsertBarrier();

  void set_compute_pipeline_state(MTL::ComputePipelineState* kernel) {
    get_command_encoder()->setComputePipelineState(kernel);
  }

  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, size_t nelems, int idx) {
    using T = typename Vec::value_type;
    if (nelems > 0) {
      get_command_encoder()->setBytes(vec.data(), nelems * sizeof(T), idx);
    } else {
      T val{};
      get_command_encoder()->setBytes(&val, sizeof(T), idx);
    }
  }
  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, int idx) {
    return set_vector_bytes(vec, vec.size(), idx);
  }

  template <typename T>
  void set_bytes(const T* v, int n, int idx) {
    return get_command_encoder()->setBytes(v, n * sizeof(T), idx);
  }

  template <typename T>
  void set_bytes(const T& v, int idx) {
    return get_command_encoder()->setBytes(&v, sizeof(T), idx);
  }

  void set_threadgroup_memory_length(size_t length, int idx) {
    get_command_encoder()->setThreadgroupMemoryLength(length, idx);
  }

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

  void barrier();
  void end_encoding();
  void wait_event(Event event, uint64_t value);
  void signal_event(Event event, uint64_t value);
  bool needs_commit() const;
  const char* commit_reason() const;
  void commit(
      std::function<void()> completion = nullptr,
      const char* reason = "unspecified");
  void synchronize();

  // MLX_METAL_COMMAND_TRACE level (0 = off); see device.cpp.
  int trace_level() const {
    return trace_level_;
  }
  void trace_primitive() {
    ++trace_primitives_;
  }

  MTL::CommandBuffer* get_command_buffer() const {
    return buffer_.get();
  }

 private:
  MTL::ComputeCommandEncoder* get_command_encoder();
  void restore_input_budget(const void* buffer);

  Device& device_;
  bool exiting_{false};

  // Buffer that stores encoded commands.
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::CommandBuffer> buffer_;
  int buffer_ops_{0};
  size_t buffer_sizes_{0};
  // Reduced first bindings in the current encoder. Any later binding, even
  // through an alias or the budget API, conservatively restores the discount.
  std::unordered_map<const void*, size_t> input_budget_discounts_;

  // The residency set and how many of its sets this queue has attached.
  ResidencySets& residency_sets_;
  uint64_t sets_attached_{0};
  // Opt-in diagnostics. These counters do not change command boundaries or
  // request unsupported per-dispatch GPU timestamp sampling.
  int trace_level_{0};
  int trace_stream_index_{0};
  size_t trace_primitives_{0};
  double trace_encode_start_{0};
  size_t trace_resource_bytes_{0};
  size_t trace_barriers_{0};
  size_t trace_encoders_{0};
  std::unordered_set<const void*> trace_resources_;
  void trace_resource(const MTL::Buffer* buffer);

  // The events hooked to current command buffer.
  std::vector<Event> wait_events_;
  std::vector<std::tuple<Event, uint64_t>> signal_events_;

  // Error from previous commited command buffer.
  Error error_;

  // Encoder for issuing GPU commands.
  // The members are used within a single ComputeCommandEncoder and will be
  // reset after calling end_encoding().
  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
  NS::SharedPtr<MTL::Fence> fence_;
  bool needs_barrier_{false};
  bool concurrent_{false};
  std::vector<array> temporaries_;
  // Resources read/written since the last barrier, stamped with the epoch
  // that recorded them: a resource counts iff its stamp equals
  // barrier_epoch_. A barrier only bumps the epoch; clearing a set per
  // dispatch zeroed its bucket array every time (7% of a MoE decode thread).
  std::unordered_map<MTL::Resource*, uint64_t> prev_inputs_;
  std::unordered_map<MTL::Resource*, uint64_t> prev_outputs_;
  uint64_t barrier_epoch_{1};
  // Resources of the kernel being encoded; maybeInsertBarrier() stamps them.
  std::vector<MTL::Resource*> next_inputs_;
  std::vector<MTL::Resource*> next_outputs_;
  std::unordered_set<MTL::Resource*> concurrent_outputs_;
  std::unordered_set<const void*> all_inputs_;
  std::unordered_set<const void*> all_outputs_;

  // Array storage retained until the next commit() — moved into a single
  // completed-handler at commit time (see retain()).
  std::unordered_set<std::shared_ptr<array::Data>> pending_retained_;

  // Test access injects a deterministic pin failure without a production
  // runtime switch or public test method. Publication remains all-or-nothing.
  friend struct PersistentWorkspaceTestAccess;
  template <typename Pin>
  void publish_workspace(
      uint64_t identity, std::vector<array> prepared, Pin&& pin) {
    try {
      for (const auto& slot : prepared) pin(slot.data_shared_ptr());
    } catch (...) {
      // None of these newly allocated slots has been bound to the GPU. Erase
      // every possibly inserted pin so repeated caught failures stay bounded.
      for (const auto& slot : prepared)
        pending_retained_.erase(slot.data_shared_ptr());
      throw;
    }
    workspace_.swap(prepared);
    workspace_identity_ = identity;
  }

  // Never resized or evicted. Completion pins also hold each acquired slot.
  uint64_t workspace_identity_{0};
  std::vector<array> workspace_;

  // A map of prior command encoder outputs to their corresponding fence.
  std::unordered_map<const void*, NS::SharedPtr<MTL::Fence>> prev_ce_outputs_;
  std::mutex outputs_mtx_;
};

class MLX_API Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  MTL::Device* mtl_device() {
    return device_.get();
  };

  const std::string& get_architecture() const {
    return arch_;
  }
  int get_architecture_gen() const {
    return arch_gen_;
  }
  std::tuple<int, int> get_max_ops_mb_per_buffer() const {
    return std::make_tuple(max_ops_per_buffer_, max_mb_per_buffer_);
  }

  MTL::Library* get_library(
      const std::string& name,
      const std::string& path = "");

  MTL::Library* get_library(
      const std::string& name,
      const CompileOptions& compile_options,
      const std::function<std::string(void)>& builder);

  MTL::Library* get_library(
      const std::string& name,
      const std::function<std::string(void)>& builder) {
    return get_library(name, {}, builder);
  }

  void clear_library(const std::string& name);

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  ResidencySets& residency_sets() {
    return residency_sets_;
  }

 private:
  NS::SharedPtr<MTL::Library> build_library_(
      const std::string& source_string,
      const CompileOptions& compile_options = {});

  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      MTL::Library* mtl_lib);
  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      const std::string& specialized_name,
      const MTLFCList& func_consts,
      MTL::Library* mtl_lib);

  NS::SharedPtr<MTL::LinkedFunctions> get_linked_functions_(
      const std::vector<MTL::Function*>& funcs);

  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function);
  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function,
      const MTL::LinkedFunctions* linked_functions);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name,
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  NS::SharedPtr<MTL::Device> device_;
  ResidencySets residency_sets_;

  std::shared_mutex kernel_mtx_;
  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::Library>> library_map_;
  NS::SharedPtr<MTL::Library> default_library_;
  std::unordered_map<
      MTL::Library*,
      std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>>
      library_kernels_;
  std::string arch_;
  int arch_gen_;
  int max_ops_per_buffer_;
  int max_mb_per_buffer_;
};

MLX_API Device& device(mlx::core::Device);
MLX_API CommandEncoder& get_command_encoder(Stream s);

std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();
NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool();

bool is_nax_available();

} // namespace mlx::core::metal
