// Copyright 2026 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  // Force initialization of the WebGPU device.
  wgpu::device();
  // Ensure the command encoder for this stream gets created.
  wgpu::get_command_encoder(s);
}

void eval(array& arr) {
  auto outputs = arr.outputs();
  {
    // If the array is a tracer, hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  auto& stream = arr.primitive().stream();
  auto& encoder = wgpu::get_command_encoder(stream);

  // Keep used buffers alive until kernel finishes running.
  for (auto& in : arr.inputs()) {
    // Except for the donated one.
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }

  if (encoder.needs_commit()) {
    scheduler::notify_new_task(stream);
    encoder.add_completed_handler(
        [stream]() { scheduler::notify_task_completion(stream); });
    encoder.commit();
  }
}

void finalize(Stream s) {
  wgpu::get_command_encoder(s).commit();
}

void synchronize(Stream s) {
  wgpu::get_command_encoder(s).synchronize();
}

} // namespace mlx::core::gpu
