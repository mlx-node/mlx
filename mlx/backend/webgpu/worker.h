// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <functional>
#include <vector>

#if !defined(__wasi__)
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>
#endif

namespace mlx::core::wgpu {

class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // DEV-F011: add_task / add_completed_handler has platform-asymmetric
  // semantics: on native builds (Dawn/wgpu-native) the task fires from
  // wgpuQueueOnSubmittedWorkDone AFTER the GPU finishes; on WASI builds
  // (no background thread, no real async callback because synchronize()
  // would deadlock on f.wait()) it fires synchronously during commit(),
  // i.e. at SUBMIT time, not at GPU-done. Callers that rely on true
  // GPU-done timing on both platforms must use a different primitive
  // (e.g. a queue fence with its own mapAsync). add_at_submit() is a
  // same-signature alias that makes that asymmetry explicit in new
  // call sites without breaking existing ones.
  void add_task(std::function<void()> task);

  // Alias for add_task(): documents that on WASI the task runs at
  // commit-time, not at GPU-done. Prefer this name when you do NOT
  // need at-completion semantics.
  void add_at_submit(std::function<void()> task) { add_task(std::move(task)); }

  void commit(WGPUQueue queue);

 private:
  using Tasks = std::vector<std::function<void()>>;
  Tasks pending_tasks_;

#if !defined(__wasi__)
  void thread_fn();

  std::mutex mtx_;
  std::condition_variable cond_;

  uint64_t committed_batch_{0};
  uint64_t signaled_batch_{0};

  bool stop_{false};

  std::map<uint64_t, Tasks> worker_tasks_;
  std::thread worker_;
#endif
};

} // namespace mlx::core::wgpu
