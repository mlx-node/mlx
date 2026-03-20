// Copyright 2026 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/scheduler.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace mlx::core {

// An event counter using atomics + condition variable for synchronization.
struct EventCounter {
  std::atomic<uint64_t> value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event_ = std::shared_ptr<void>(new EventCounter{}, dtor);
}

void Event::wait() {
  auto* ec = static_cast<EventCounter*>(event_.get());
  uint64_t target = value();
  if (ec->value.load(std::memory_order_acquire) >= target) {
    return;
  }
  std::unique_lock<std::mutex> lk(ec->mtx);
  ec->cv.wait(lk, [ec, target] {
    return ec->value.load(std::memory_order_acquire) >= target;
  });
}

void Event::wait(Stream s) {
  if (s.device == Device::cpu) {
    scheduler::enqueue(s, [*this]() mutable { wait(); });
  } else {
    // GPU stream: synchronize the encoder, then wait on CPU
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.synchronize();
    wait();
  }
}

void Event::signal(Stream s) {
  auto* ec = static_cast<EventCounter*>(event_.get());
  uint64_t target = value();

  if (s.device == Device::cpu) {
    scheduler::enqueue(s, [ec, target]() {
      ec->value.store(target, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lk(ec->mtx);
      }
      ec->cv.notify_all();
    });
  } else {
    // GPU stream: commit pending work, then signal after GPU completes
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.commit();
    encoder.add_completed_handler([ec, target]() {
      ec->value.store(target, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lk(ec->mtx);
      }
      ec->cv.notify_all();
    });
    encoder.commit();
  }
}

bool Event::is_signaled() const {
  auto* ec = static_cast<EventCounter*>(event_.get());
  return ec->value.load(std::memory_order_acquire) >= value();
}

} // namespace mlx::core
