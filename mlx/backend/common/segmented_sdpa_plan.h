// Copyright © 2026

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlx::core::fast {

struct SegmentedSdpaCapabilities {
  size_t thread_execution_width;
  size_t max_threads_per_threadgroup;
  size_t static_threadgroup_memory;
  size_t max_threadgroup_memory;
};

struct SegmentedSdpaLaunchPlan {
  bool supported;
  bool two_pass;
  uint32_t partitions;
  uint32_t stage1_threads;
  uint32_t stage2_threads;
};

// Pure planner shared by the Metal dispatch and platform-independent tests.
// D=256 and the 32-lane reduction are algorithm shape constants; all device
// limits are supplied by the selected pipelines and MTLDevice at runtime.
inline SegmentedSdpaLaunchPlan plan_segmented_sdpa_launch(
    int query_length,
    int gqa_factor,
    bool two_pass,
    int partitions,
    const SegmentedSdpaCapabilities& stage1,
    const SegmentedSdpaCapabilities* stage2) {
  SegmentedSdpaLaunchPlan plan{
      false, two_pass, static_cast<uint32_t>(std::max(partitions, 0)), 0, 0};
  if (query_length < 1 || query_length > 8 || gqa_factor < 1 ||
      gqa_factor > 32 || partitions < 32 || (partitions % 32) != 0 ||
      stage1.thread_execution_width != 32 ||
      stage1.static_threadgroup_memory > stage1.max_threadgroup_memory) {
    return plan;
  }
  if (two_pass) {
    const uint64_t stage1_threads = uint64_t{32} * gqa_factor * query_length;
    if (stage1_threads > stage1.max_threads_per_threadgroup ||
        stage2 == nullptr || stage2->thread_execution_width != 32 ||
        stage2->max_threads_per_threadgroup < 1024 ||
        stage2->static_threadgroup_memory > stage2->max_threadgroup_memory) {
      return plan;
    }
    plan.stage1_threads = static_cast<uint32_t>(stage1_threads);
    plan.stage2_threads = 1024;
  } else {
    if (stage1.max_threads_per_threadgroup < 1024) {
      return plan;
    }
    plan.stage1_threads = 1024;
  }
  plan.supported = true;
  return plan;
}

} // namespace mlx::core::fast
