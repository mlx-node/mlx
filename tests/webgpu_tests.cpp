// Copyright © 2024 Apple Inc.
//
// WebGPU-backend correctness tests. Built only when MLX_BUILD_WEBGPU=ON.
//
// These target the `scatter_add` (Scatter with Sum reduce) kernel, which is the
// embedding-table gradient (`Gather::vjp`). The kernel must ACCUMULATE when two
// update elements map to the same output cell (a token id repeated in a
// sequence). A last-write-wins kernel silently undercounts the gradient, so we
// pin the behavior against the CPU reference with deliberately colliding indices.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("webgpu device sanity") {
  // Confirms the native WebGPU device initializes and runs basic ops, so a
  // scatter failure below cannot be mistaken for a broken device.
  auto a = arange(0, 8, 1, float32, Device::gpu);
  auto b = arange(0, 8, 1, float32, Device::cpu);
  CHECK(array_equal(a, b, Device::cpu).item<bool>());
  auto c = add(a, a, Device::gpu);
  CHECK(array_equal(c, add(b, b, Device::cpu), Device::cpu).item<bool>());
}

TEST_CASE("webgpu scatter_add 1d collision") {
  // Index 0 appears twice -> both updates must sum into out[0].
  auto in = zeros({3}, float32, Device::cpu);
  auto inds = array({0, 0, 1}, {3});
  auto upd = ones({3, 1}, float32);

  auto gpu = scatter_add(in, inds, upd, 0, Device::gpu);
  auto cpu = scatter_add(in, inds, upd, 0, Device::cpu);
  CHECK(array_equal(gpu, cpu, Device::cpu).item<bool>());
  CHECK(array_equal(gpu, array({2.0f, 1.0f, 0.0f}), Device::cpu).item<bool>());
}

TEST_CASE("webgpu scatter_add heavy collision") {
  // Ten updates all target index 3 -> out[3] == 10.
  auto in = zeros({5}, float32, Device::cpu);
  auto inds = broadcast_to(array(3), {10});
  auto upd = ones({10, 1}, float32);

  auto gpu = scatter_add(in, inds, upd, 0, Device::gpu);
  auto cpu = scatter_add(in, inds, upd, 0, Device::cpu);
  CHECK(array_equal(gpu, cpu, Device::cpu).item<bool>());
  CHECK(
      array_equal(gpu, array({0.0f, 0.0f, 0.0f, 10.0f, 0.0f}), Device::cpu)
          .item<bool>());
}

TEST_CASE("webgpu scatter_add embedding-shaped collision") {
  // Embedding-gradient shape: table [V=4, d=3]; token ids [0,2,0] each add a
  // row of ones -> row 0 accumulates to 2, row 2 to 1, rows 1 and 3 stay 0.
  auto in = zeros({4, 3}, float32, Device::cpu);
  auto inds = array({0, 2, 0}, {3});
  auto upd = ones({3, 3}, float32);

  auto gpu = scatter_add(in, inds, upd, 0, Device::gpu);
  auto cpu = scatter_add(in, inds, upd, 0, Device::cpu);
  CHECK(array_equal(gpu, cpu, Device::cpu).item<bool>());
}

TEST_CASE("webgpu scatter_add no-collision matches cpu") {
  // Distinct indices: even a last-write-wins kernel is correct here, so this
  // guards against the fix regressing the common (non-colliding) path.
  auto in = zeros({4}, float32, Device::cpu);
  auto inds = array({3, 1}, {2});
  auto upd = array({5.0f, 7.0f}, {2, 1});

  auto gpu = scatter_add(in, inds, upd, 0, Device::gpu);
  auto cpu = scatter_add(in, inds, upd, 0, Device::cpu);
  CHECK(array_equal(gpu, cpu, Device::cpu).item<bool>());
}
