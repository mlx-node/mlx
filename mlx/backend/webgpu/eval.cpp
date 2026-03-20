// Copyright © 2026 Apple Inc.

#include <stdexcept>

#include "mlx/backend/gpu/eval.h"

namespace mlx::core::gpu {

void new_stream(Stream) {}

void eval(array&) {
  throw std::runtime_error(
      "[gpu::eval] WebGPU backend is not initialized");
}

void finalize(Stream) {
  throw std::runtime_error(
      "[gpu::finalize] WebGPU backend is not initialized");
}

void synchronize(Stream) {
  throw std::runtime_error(
      "[gpu::synchronize] WebGPU backend is not initialized");
}

} // namespace mlx::core::gpu
