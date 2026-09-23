// Copyright © 2026

#pragma once

#include <Metal/Metal.hpp>

#include "mlx/backend/common/segmented_sdpa_plan.h"
#include "mlx/backend/metal/device.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

// These two functions are the vector-SDPA reduction policy.  The segmented
// kernel shares them with the contiguous baseline because changing either the
// route or partition count changes the floating-point reduction order.
inline bool sdpa_vector_uses_two_pass(
    const metal::Device& device,
    int sequence_length,
    int query_heads,
    int kv_heads) {
  const char device_class = device.get_architecture().back();
  return ((device_class == 'd' || device_class == 's') &&
          sequence_length >= 1024) ||
      (kv_heads < query_heads && sequence_length >= 4096);
}

inline int sdpa_vector_partition_count(
    const metal::Device& device,
    int sequence_length,
    int active_simdgroups) {
  const char device_class = device.get_architecture().back();
  int partitions;
  if (device_class == 's') {
    partitions = 64;
    if (sequence_length > 1024 && active_simdgroups > 4) {
      if (sequence_length <= 8192) {
        partitions = 128;
      } else if (sequence_length <= 32768) {
        partitions = 256;
      } else if (sequence_length <= 65536) {
        partitions = 512;
      } else {
        partitions = 1024;
      }
    }
  } else if (device_class == 'd') {
    partitions = 128;
    if (active_simdgroups <= 2 && sequence_length > 8192) {
      partitions = 256;
    } else if (active_simdgroups >= 6) {
      if (sequence_length >= 16384 && sequence_length < 65536) {
        partitions = 512;
      } else if (sequence_length >= 65536) {
        partitions = 1024;
      }
    }
  } else {
    partitions = active_simdgroups >= 4 ? 64 : 32;
  }
  if (int override = env::get_var("MLX_SDPA_BLOCKS", 0); override > 0) {
    // The aggregation kernel consumes partitions in 32-wide chunks.
    partitions = ((override + 31) / 32) * 32;
  }
  return partitions;
}

SegmentedSdpaCapabilities segmented_sdpa_capabilities(
    MTL::ComputePipelineState* pipeline,
    MTL::Device* device);

} // namespace mlx::core::fast
