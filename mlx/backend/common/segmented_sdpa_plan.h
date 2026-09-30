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

struct SegmentedSdpaReductionPlan {
  bool two_pass;
  int partitions;
};

enum class SegmentedVerifyRoute : int {
  single = 0,
  one_pass = 1,
  unified = 2,
  split = 3,
};

// Rows of the leading chunk when a verify block exceeds the widest supported
// query chunk: the trailing chunk takes as many rows as the device allows.
// 0 = one chunk; -1 = two chunks cannot cover the block.
inline int segmented_verify_head_rows(int rows, int max_query_length) {
  if (rows < 1 || rows > 8 || max_query_length < 1) {
    return -1;
  }
  if (rows <= max_query_length) {
    return 0;
  }
  const int head = rows - std::min(max_query_length, rows - 1);
  return head > max_query_length ? -1 : head;
}

// First pass of the one-call verify kernel: two (head, row) pairs per
// simdgroup, all pairs of one KV head in one threadgroup.
inline SegmentedSdpaLaunchPlan plan_segmented_verify_launch(
    int rows,
    int gqa_factor,
    int partitions,
    const SegmentedSdpaCapabilities& stage1,
    const SegmentedSdpaCapabilities& stage2) {
  SegmentedSdpaLaunchPlan plan{
      false, true, static_cast<uint32_t>(std::max(partitions, 0)), 0, 0};
  const int64_t pairs = int64_t{rows} * gqa_factor;
  if (rows < 2 || rows > 8 || gqa_factor < 1 || gqa_factor > 32 ||
      pairs % 2 != 0 || partitions < 32 || (partitions % 32) != 0 ||
      stage1.thread_execution_width != 32 ||
      stage1.static_threadgroup_memory > stage1.max_threadgroup_memory ||
      stage2.thread_execution_width != 32 ||
      stage2.max_threads_per_threadgroup < 1024 ||
      stage2.static_threadgroup_memory > stage2.max_threadgroup_memory) {
    return plan;
  }
  const uint64_t stage1_threads = uint64_t{32} * uint64_t(pairs / 2);
  if (stage1_threads > stage1.max_threads_per_threadgroup) {
    return plan;
  }
  plan.stage1_threads = static_cast<uint32_t>(stage1_threads);
  plan.stage2_threads = 1024;
  plan.supported = true;
  return plan;
}

// The reduction order of a (head, row) pair depends only on the route and
// partition count of the chunk that owns the row. One dispatch over the whole
// block is therefore exact only when both chunks reduce the same way.
// Requires head_rows >= 0.
inline SegmentedVerifyRoute select_segmented_verify_route(
    int head_rows,
    SegmentedSdpaReductionPlan head,
    SegmentedSdpaReductionPlan tail,
    bool unified_supported) {
  if (head_rows == 0) {
    return SegmentedVerifyRoute::single;
  }
  if (!head.two_pass && !tail.two_pass) {
    return SegmentedVerifyRoute::one_pass;
  }
  if (head.two_pass && tail.two_pass && head.partitions == tail.partitions &&
      unified_supported) {
    return SegmentedVerifyRoute::unified;
  }
  return SegmentedVerifyRoute::split;
}

} // namespace mlx::core::fast
