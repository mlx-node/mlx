// Copyright 2026 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/scheduler.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace mlx::core {

struct FenceImpl {
  uint32_t count{0};
  std::atomic<uint32_t> value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

Fence::Fence(Stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
}

void Fence::wait(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  uint32_t target = fence->count;

  if (s.device == Device::cpu) {
    scheduler::enqueue(s, [target, fence_ptr = fence_]() {
      auto* f = static_cast<FenceImpl*>(fence_ptr.get());
      if (f->value.load(std::memory_order_acquire) >= target) {
        return;
      }
      std::unique_lock<std::mutex> lk(f->mtx);
      f->cv.wait(lk, [f, target] {
        return f->value.load(std::memory_order_acquire) >= target;
      });
    });
  } else {
    // GPU stream: synchronize, then wait
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.synchronize();
    if (fence->value.load(std::memory_order_acquire) >= target) {
      return;
    }
    std::unique_lock<std::mutex> lk(fence->mtx);
    fence->cv.wait(lk, [fence, target] {
      return fence->value.load(std::memory_order_acquire) >= target;
    });
  }
}

void Fence::update(Stream s, const array&, bool /* cross_device */) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  uint32_t target = fence->count;

  if (s.device == Device::cpu) {
    scheduler::enqueue(s, [target, fence_ptr = fence_]() {
      auto* f = static_cast<FenceImpl*>(fence_ptr.get());
      f->value.store(target, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lk(f->mtx);
      }
      f->cv.notify_all();
    });
  } else {
    // GPU stream: commit pending work, signal after GPU completes
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.commit();
    encoder.add_completed_handler([target, fence_ptr = fence_]() {
      auto* f = static_cast<FenceImpl*>(fence_ptr.get());
      f->value.store(target, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lk(f->mtx);
      }
      f->cv.notify_all();
    });
    encoder.commit();
  }
}

} // namespace mlx::core
