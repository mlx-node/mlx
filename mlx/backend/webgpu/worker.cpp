// Copyright 2026 Apple Inc.

#include "mlx/backend/webgpu/worker.h"

#include <cstdio>

namespace mlx::core::wgpu {

#if defined(__wasi__)

// WASI: no background thread. Run tasks synchronously during commit()
// and send wgpuQueueOnSubmittedWorkDone for gpu-worker lifecycle.
// Tasks MUST run inline because synchronize() blocks with f.wait()
// which prevents POLL from firing the async callback.
Worker::Worker() {}
Worker::~Worker() {}

void Worker::add_task(std::function<void()> task) {
  pending_tasks_.push_back(std::move(task));
}

void Worker::commit(WGPUQueue queue) {
  // Run tasks synchronously — can't defer to callback because
  // synchronize() blocks on f.wait() preventing POLL from running.
  Tasks tasks;
  tasks.swap(pending_tasks_);
  for (auto& task : tasks) {
    try {
      task();
    } catch (const std::exception& e) {
      fprintf(stderr, "[WebGPU] Exception in completion handler: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[WebGPU] Unknown exception in completion handler\n");
    }
  }

  // Still notify gpu-worker that work was submitted (for its lifecycle)
  auto noop = [](WGPUQueueWorkDoneStatus, void*) {};
  wgpuQueueOnSubmittedWorkDone(queue, noop, nullptr);
}

#else

Worker::Worker() : worker_(&Worker::thread_fn, this) {}

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_ = true;
  }
  cond_.notify_one();
  worker_.join();
}

void Worker::add_task(std::function<void()> task) {
  pending_tasks_.push_back(std::move(task));
}

void Worker::commit(WGPUQueue queue) {
  if (pending_tasks_.empty()) {
    return;
  }

  uint64_t batch;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    batch = ++committed_batch_;
    worker_tasks_[batch] = std::move(pending_tasks_);
  }

  struct CallbackData {
    Worker* worker;
    uint64_t batch;
  };
  auto* cb_data = new CallbackData{this, batch};

  auto callback = [](WGPUQueueWorkDoneStatus status, void* userdata) {
    if (status != WGPUQueueWorkDoneStatus_Success) {
      fprintf(stderr, "[WebGPU] Queue work done failed (status=%d)\n",
              static_cast<int>(status));
    }
    auto* data = static_cast<CallbackData*>(userdata);
    {
      std::lock_guard<std::mutex> lock(data->worker->mtx_);
      if (data->batch > data->worker->signaled_batch_) {
        data->worker->signaled_batch_ = data->batch;
      }
    }
    data->worker->cond_.notify_one();
    delete data;
  };

#if defined(WEBGPU_BACKEND_WGPU)
  wgpuQueueOnSubmittedWorkDone(queue, callback, cb_data);
#else
  wgpuQueueOnSubmittedWorkDone(queue, 0u, callback, cb_data);
#endif
}

void Worker::thread_fn() {
  uint64_t current_batch = 0;
  while (true) {
    Tasks tasks;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cond_.wait(lk, [this, &current_batch] {
        return signaled_batch_ > current_batch || stop_;
      });
      if (stop_ && worker_tasks_.empty()) {
        return;
      }
      current_batch = signaled_batch_;
      auto end = worker_tasks_.upper_bound(current_batch);
      for (auto it = worker_tasks_.begin(); it != end; ++it) {
        if (tasks.empty()) {
          tasks = std::move(it->second);
        } else {
          std::move(it->second.begin(), it->second.end(), std::back_inserter(tasks));
        }
      }
      worker_tasks_.erase(worker_tasks_.begin(), end);
    }
    for (size_t i = 0; i < tasks.size(); ++i) {
      auto task = std::move(tasks[i]);
      try {
        task();
      } catch (const std::exception& e) {
        fprintf(stderr, "[WebGPU] Exception in completion handler: %s\n", e.what());
      } catch (...) {
        fprintf(stderr, "[WebGPU] Unknown exception in completion handler\n");
      }
    }
  }
}

#endif

} // namespace mlx::core::wgpu
