// Copyright © 2026

#pragma once

// Vector SDPA over two logically adjacent K/V segments.  The score traversal
// and online-softmax update order intentionally match sdpa_vector.h; only the
// address calculation chooses prefix or newly-produced K/V.

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_segmented(
    const device T* queries [[buffer(0)]],
    const device T* prefix_keys [[buffer(1)]],
    const device T* prefix_values [[buffer(2)]],
    const device T* new_keys [[buffer(3)]],
    const device T* new_values [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& gqa_factor [[buffer(6)]],
    const constant int& prefix_n [[buffer(7)]],
    const constant int& new_n [[buffer(8)]],
    const constant long* strides [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const constant int& num_q_heads [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  // q batch/head/sequence; then prefix K/V and new K/V batch/head/sequence.
  const long q_batch_stride = strides[0];
  const long q_head_stride = strides[1];
  const long q_seq_stride = strides[2];
  const long pk_batch_stride = strides[3];
  const long pk_head_stride = strides[4];
  const long pk_seq_stride = strides[5];
  const long pv_batch_stride = strides[6];
  const long pv_head_stride = strides[7];
  const long pv_seq_stride = strides[8];
  const long nk_batch_stride = strides[9];
  const long nk_head_stride = strides[10];
  const long nk_seq_stride = strides[11];
  const long nv_batch_stride = strides[12];
  const long nv_head_stride = strides[13];
  const long nv_seq_stride = strides[14];

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];
  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int batch_idx = q_batch_head_idx / num_q_heads;
  const int q_head_idx = q_batch_head_idx - batch_idx * num_q_heads;
  const int kv_head_idx = q_head_idx / gqa_factor;
  const int n = prefix_n + new_n;
  queries += batch_idx * q_batch_stride + q_head_idx * q_head_stride +
      q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  out += (q_batch_head_idx * tpg.y + q_seq_idx) * V + simd_gid * v_per_thread;

  for (int j = 0; j < qk_per_thread; ++j) {
    q[j] = static_cast<U>(scale) * queries[j];
  }
  for (int j = 0; j < v_per_thread; ++j) {
    o[j] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  // Every prefix row precedes every query. Traverse that segment without
  // a per-row segment selection or causal branch, then continue the same
  // strided score sequence through the visible new rows.
  const device T* pk = prefix_keys + batch_idx * pk_batch_stride +
      kv_head_idx * pk_head_stride + simd_lid * qk_per_thread;
  const device T* pv = prefix_values + batch_idx * pv_batch_stride +
      kv_head_idx * pv_head_stride + simd_lid * v_per_thread;
  int i = simd_gid;
  for (; i < prefix_n; i += BN) {
    const device T* key = pk + i * pk_seq_stride;
    const device T* value = pv + i * pv_seq_stride;
    for (int j = 0; j < qk_per_thread; ++j) {
      k[j] = key[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
  }
  const device T* nk = new_keys + batch_idx * nk_batch_stride +
      kv_head_idx * nk_head_stride + simd_lid * qk_per_thread;
  const device T* nv = new_values + batch_idx * nv_batch_stride +
      kv_head_idx * nv_head_stride + simd_lid * v_per_thread;
  const int visible_n = do_causal ? n - int(tpg.y) + q_seq_idx + 1 : n;
  for (; i < visible_n; i += BN) {
    const device T* key = nk + (i - prefix_n) * nk_seq_stride;
    const device T* value = nv + (i - prefix_n) * nv_seq_stride;
    for (int j = 0; j < qk_per_thread; ++j) {
      k[j] = key[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int j = 0; j < v_per_thread; ++j) {
    outputs[simd_lid * BD + simd_gid] = o[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[j] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[j] = sum_exp_score == 0 ? o[j] : (o[j] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int j = 0; j < v_per_thread; ++j) {
      out[j] = static_cast<T>(o[j]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_segmented_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* prefix_keys [[buffer(1)]],
    const device T* prefix_values [[buffer(2)]],
    const device T* new_keys [[buffer(3)]],
    const device T* new_values [[buffer(4)]],
    device T* out [[buffer(5)]],
    device float* sums [[buffer(6)]],
    device float* maxs [[buffer(7)]],
    const constant int& prefix_n [[buffer(8)]],
    const constant int& new_n [[buffer(9)]],
    const constant long* strides [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_q_heads = tpg.x * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int n = prefix_n + new_n;

  queries += batch_idx * strides[0] + q_head_idx * strides[1] +
      q_seq_idx * strides[2] + simd_lid * qk_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  for (int j = 0; j < qk_per_thread; ++j) {
    q[j] = static_cast<U>(scale) * queries[j];
  }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  const device T* pk = prefix_keys + batch_idx * strides[3] +
      kv_head_idx * strides[4] + simd_lid * qk_per_thread;
  const device T* pv = prefix_values + batch_idx * strides[6] +
      kv_head_idx * strides[7] + simd_lid * v_per_thread;
  int i = block_idx;
  for (; i < prefix_n; i += blocks) {
    const device T* key = pk + i * strides[5];
    const device T* value = pv + i * strides[8];
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * key[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
  }
  const device T* nk = new_keys + batch_idx * strides[9] +
      kv_head_idx * strides[10] + simd_lid * qk_per_thread;
  const device T* nv = new_values + batch_idx * strides[12] +
      kv_head_idx * strides[13] + simd_lid * v_per_thread;
  const int visible_n = do_causal ? n - q_seq_len + q_seq_idx + 1 : n;
  for (; i < visible_n; i += blocks) {
    const device T* key = nk + (i - prefix_n) * strides[11];
    const device T* value = nv + (i - prefix_n) * strides[14];
    U score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
      score += q[j] * key[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; ++j) {
      o[j] = o[j] * factor + exp_score * value[j];
    }
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int j = 0; j < v_per_thread; ++j) {
    out[j] = static_cast<T>(o[j]);
  }
}
