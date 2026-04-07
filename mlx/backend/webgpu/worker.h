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

// Run tasks in a worker thread, synchronized with WebGPU queue completion.
// On WASI, tasks run synchronously during commit (no background thread).
class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Add a pending task that will run when committed and GPU work finishes.
  void add_task(std::function<void()> task);

  // Batch pending tasks and register a wgpuQueueOnSubmittedWorkDone callback.
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
