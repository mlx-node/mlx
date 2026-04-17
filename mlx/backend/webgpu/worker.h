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

  void add_task(std::function<void()> task);
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
