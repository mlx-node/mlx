// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/atomic.h"

using namespace metal;

// Note: Function constants (has_mask, query_transposed, do_causal, bool_mask,
// float_mask, has_sinks) are defined in sdpa_vector.h with indices 20-25.
// This header assumes sdpa_vector.h is included first.

///////////////////////////////////////////////////////////////////////////////
// SDPA Vector VJP Kernel
//
// Computes gradients dQ, dK, dV for scaled dot-product attention backward pass.
//
// Forward: O = softmax(scale * Q @ K^T) @ V
//
// Backward (VJP):
//   P = softmax(scale * Q @ K^T)  [reconstructed from logsumexp]
//   dV = P^T @ dO
//   dP = dO @ V^T
//   dS = P * (dP - sum(dP * P))  [softmax gradient]
//   dQ = scale * dS @ K
//   dK = scale * dS^T @ Q
//
// This kernel handles the "vector" case where Q_seq is small (<=8).
// Each threadgroup processes one (batch, head, q_seq) position.
//
// IMPORTANT: Stride Assumption
// This kernel uses input strides (k_head_stride, k_seq_stride, v_head_stride,
// v_seq_stride) for output array (d_keys, d_values) pointer arithmetic.
// The dispatch code must ensure that output arrays have matching strides:
//   - d_k.strides() must match k.strides() for head and seq dimensions
//   - d_v.strides() must match v.strides() for head and seq dimensions
// Failure to maintain this invariant will cause memory corruption.
///////////////////////////////////////////////////////////////////////////////

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_vjp(
    // Forward inputs
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    // Forward output and upstream gradient
    const device T* out [[buffer(3)]],
    const device T* d_out [[buffer(4)]],
    // Logsumexp from forward (for numerically stable softmax reconstruction)
    const device float* logsumexp [[buffer(5)]],
    // Gradient outputs
    device T* d_queries [[buffer(6)]],
    device T* d_keys [[buffer(7)]],
    device T* d_values [[buffer(8)]],
    // Attention parameters
    const constant int& gqa_factor [[buffer(9)]],
    const constant int& N [[buffer(10)]],  // KV sequence length
    const constant size_t& k_head_stride [[buffer(11)]],
    const constant size_t& k_seq_stride [[buffer(12)]],
    const constant size_t& v_head_stride [[buffer(13)]],
    const constant size_t& v_seq_stride [[buffer(14)]],
    const constant float& scale [[buffer(15)]],
    // Optional mask inputs
    const device bool* bmask [[buffer(16), function_constant(bool_mask)]],
    const device T* fmask [[buffer(17), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
        [[buffer(18), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
        [[buffer(19), function_constant(has_mask)]],
    const constant int& mask_head_stride
        [[buffer(20), function_constant(has_mask)]],
    // Optional attention sinks
    const device T* sinks [[buffer(21), function_constant(has_sinks)]],
    const constant int& num_q_heads
        [[buffer(22), function_constant(has_sinks)]],
    // Thread position info
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  // Block sizes matching forward kernel
  constexpr int BN = 32;  // Number of simdgroups (parallel KV positions)
  constexpr int BD = 32;  // Simdgroup width (threads per simdgroup)
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;  // Accumulator type for numerical stability

  // Thread-local storage for queries, keys, values, gradients
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  thread U dq[qk_per_thread];  // Gradient w.r.t. query
  thread U d_o[v_per_thread];  // Upstream gradient
  thread U o[v_per_thread];    // Forward output (for delta computation)

  // Threadgroup memory for reductions and communication
  threadgroup U shared_delta[1];         // delta = sum(dO * O)
  threadgroup U shared_dQ[BN * D];       // For dQ reduction across simdgroups

  // Compute positions (same as forward)
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  // Set up input pointers
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;

  // Set up mask pointers if needed
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  // Set up output/gradient pointers
  out += o_offset * V + simd_lid * v_per_thread;
  d_out += o_offset * V + simd_lid * v_per_thread;
  logsumexp += o_offset;

  d_queries += q_offset * D + simd_lid * qk_per_thread;
  d_keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  d_values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;

  // Load query (scaled)
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  // Initialize dQ accumulator to zero
  for (int i = 0; i < qk_per_thread; i++) {
    dq[i] = 0;
  }

  // Load forward output O and upstream gradient dO
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = out[i];
    d_o[i] = d_out[i];
  }

  // Compute delta = sum(dO * O) - needed for softmax gradient
  // This is invariant across all KV positions for this query
  U local_delta = 0;
  for (int i = 0; i < v_per_thread; i++) {
    local_delta += d_o[i] * o[i];
  }
  // Sum across simdgroup
  local_delta = simd_sum(local_delta);

  // First simdgroup stores delta to shared memory
  if (simd_gid == 0 && simd_lid == 0) {
    shared_delta[0] = local_delta;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U delta = shared_delta[0];

  // Load logsumexp for this query position
  U lse = logsumexp[0];

  // Initialize shared_dQ to zero
  for (int idx = simd_gid * BD + simd_lid; idx < BN * D; idx += BN * BD) {
    shared_dQ[idx] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Main loop over KV sequence
  for (int kv_idx = simd_gid; kv_idx < N; kv_idx += BN) {
    bool use_key = true;

    // Apply causal or explicit mask
    if (do_causal) {
      use_key = kv_idx <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }

    if (use_key) {
      // Load key for this position
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Load value for this position
      for (int j = 0; j < v_per_thread; j++) {
        v[j] = values[j];
      }

      // Reconstruct attention score: S = scale * Q @ K^T
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Add float mask if present
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Reconstruct attention probability: P = 2^(S - logsumexp)
      // Using exp2 because forward kernel computes LSE in log base 2 domain
      U prob = fast::exp2(score - lse);

      // Compute dP = dO @ V^T for this KV position
      U dP = 0;
      for (int j = 0; j < v_per_thread; j++) {
        dP += d_o[j] * v[j];
      }
      dP = simd_sum(dP);

      // Compute dS = P * (dP - delta) [softmax gradient]
      U dS = prob * (dP - delta);

      // Accumulate dQ += scale * dS @ K
      for (int j = 0; j < qk_per_thread; j++) {
        dq[j] += static_cast<U>(scale) * dS * k[j];
      }

      // Compute dK = scale * dS * Q (this position)
      // Each thread computes its portion of dK
      // Note: q[j] already has scale applied (q[j] = scale * Q[j]), so:
      // dK = scale * dS * Q = scale * dS * (q[j] / scale) = dS * q[j]
      for (int j = 0; j < qk_per_thread; j++) {
        U dk_val = dS * q[j];
        // Atomic add - multiple query positions may contribute to same dK
        mlx_atomic_fetch_add_explicit(
            reinterpret_cast<device mlx_atomic<float>*>(d_keys),
            static_cast<float>(dk_val),
            j);
      }

      // Accumulate dV += P^T @ dO = P * dO (for this KV position)
      // prob is scalar for this (q, kv) pair, broadcast to all dO elements
      // Atomic add - multiple query positions may contribute to same dV
      for (int j = 0; j < v_per_thread; j++) {
        mlx_atomic_fetch_add_explicit(
            reinterpret_cast<device mlx_atomic<float>*>(d_values),
            static_cast<float>(prob * d_o[j]),
            j);
      }
    }

    // Move to next KV block
    keys += inner_k_stride;
    values += inner_v_stride;
    d_keys += inner_k_stride;
    d_values += inner_v_stride;

    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Write accumulated dQ gradient
  // Need to reduce across simdgroups since each processed different KV positions
  // but they all contribute to the same query gradient

  // Store each simdgroup's partial dQ to shared memory
  // NOTE: Use D (head dimension) not BD (simdgroup width) for the stride
  // Each simdgroup needs D elements to store its full dQ contribution
  for (int i = 0; i < qk_per_thread; i++) {
    shared_dQ[simd_gid * D + simd_lid * qk_per_thread + i] = dq[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce dQ across simdgroups
  if (simd_gid == 0) {
    for (int i = 0; i < qk_per_thread; i++) {
      U sum_dq = 0;
      for (int sg = 0; sg < BN; sg++) {
        sum_dq += shared_dQ[sg * D + simd_lid * qk_per_thread + i];
      }
      d_queries[i] = static_cast<T>(sum_dq);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// SDPA Vector VJP Kernel - Separate dK/dV accumulation version
//
// This version is more suitable when multiple query positions exist,
// as dK and dV need proper accumulation across all query contributions.
///////////////////////////////////////////////////////////////////////////////

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_vjp_accumulate(
    // Forward inputs
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    // Forward output and upstream gradient
    const device T* out [[buffer(3)]],
    const device T* d_out [[buffer(4)]],
    // Logsumexp from forward
    const device float* logsumexp [[buffer(5)]],
    // Gradient outputs (dK and dV are accumulated atomically)
    device T* d_queries [[buffer(6)]],
    device float* d_keys_accum [[buffer(7)]],   // float for atomic accumulation
    device float* d_values_accum [[buffer(8)]], // float for atomic accumulation
    // Attention parameters
    const constant int& gqa_factor [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant int& Q_seq [[buffer(11)]],  // Number of query positions
    const constant size_t& k_head_stride [[buffer(12)]],
    const constant size_t& k_seq_stride [[buffer(13)]],
    const constant size_t& v_head_stride [[buffer(14)]],
    const constant size_t& v_seq_stride [[buffer(15)]],
    const constant float& scale [[buffer(16)]],
    // Optional mask inputs
    const device bool* bmask [[buffer(17), function_constant(bool_mask)]],
    const device T* fmask [[buffer(18), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
        [[buffer(19), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
        [[buffer(20), function_constant(has_mask)]],
    const constant int& mask_head_stride
        [[buffer(21), function_constant(has_mask)]],
    // Thread position info
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  thread U dq[qk_per_thread];
  thread U d_o[v_per_thread];
  thread U o[v_per_thread];

  threadgroup U shared_delta[1];
  threadgroup U shared_dQ[BN * D];   // For dQ reduction across simdgroups

  // Position setup
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  // Input pointer setup
  const device T* q_ptr = queries + q_offset * D + simd_lid * qk_per_thread;
  const device T* k_base = keys + kv_head_idx * k_head_stride + simd_lid * qk_per_thread;
  const device T* v_base = values + kv_head_idx * v_head_stride + simd_lid * v_per_thread;

  const device T* o_ptr = out + o_offset * V + simd_lid * v_per_thread;
  const device T* do_ptr = d_out + o_offset * V + simd_lid * v_per_thread;
  U lse = logsumexp[o_offset];

  // Output pointer setup
  device T* dq_ptr = d_queries + q_offset * D + simd_lid * qk_per_thread;
  device float* dk_base = d_keys_accum + kv_head_idx * k_head_stride + simd_lid * qk_per_thread;
  device float* dv_base = d_values_accum + kv_head_idx * v_head_stride + simd_lid * v_per_thread;

  // Mask pointer setup
  const device bool* bm_ptr = bool_mask ?
      bmask + q_batch_head_idx * mask_head_stride + q_seq_idx * mask_q_seq_stride : nullptr;
  const device T* fm_ptr = float_mask ?
      fmask + q_batch_head_idx * mask_head_stride + q_seq_idx * mask_q_seq_stride : nullptr;

  // Load query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * q_ptr[i];
    dq[i] = 0;
  }

  // Load O and dO, compute delta
  U local_delta = 0;
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = o_ptr[i];
    d_o[i] = do_ptr[i];
    local_delta += d_o[i] * o[i];
  }
  local_delta = simd_sum(local_delta);

  if (simd_gid == 0 && simd_lid == 0) {
    shared_delta[0] = local_delta;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U delta = shared_delta[0];

  // Process KV sequence
  const device T* k_ptr = k_base + simd_gid * k_seq_stride;
  const device T* v_ptr = v_base + simd_gid * v_seq_stride;
  device float* dk_ptr = dk_base + simd_gid * k_seq_stride;
  device float* dv_ptr = dv_base + simd_gid * v_seq_stride;
  int mask_offset = simd_gid * mask_kv_seq_stride;

  for (int kv_idx = simd_gid; kv_idx < N; kv_idx += BN) {
    bool use_key = true;

    if (do_causal) {
      use_key = kv_idx <= (N - Q_seq + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bm_ptr[mask_offset];
    } else if (float_mask) {
      use_key = (fm_ptr[mask_offset] >= Limits<T>::finite_min);
    }

    if (use_key) {
      // Load K, V
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = k_ptr[j];
      }
      for (int j = 0; j < v_per_thread; j++) {
        v[j] = v_ptr[j];
      }

      // Compute score and probability
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += static_cast<U>(fm_ptr[mask_offset]);
      }

      // Reconstruct probability using exp2 (forward uses log2 domain)
      U prob = fast::exp2(score - lse);

      // Compute dP
      U dP = 0;
      for (int j = 0; j < v_per_thread; j++) {
        dP += d_o[j] * v[j];
      }
      dP = simd_sum(dP);

      // Compute dS
      U dS = prob * (dP - delta);

      // Accumulate dQ
      for (int j = 0; j < qk_per_thread; j++) {
        dq[j] += static_cast<U>(scale) * dS * k[j];
      }

      // Atomic add to dK
      // dK = scale * dS * Q (q_ptr has unscaled query)
      // All threads in simdgroup contribute to different elements
      for (int j = 0; j < qk_per_thread; j++) {
        U dk_val = static_cast<U>(scale) * dS * q_ptr[j];
        mlx_atomic_fetch_add_explicit(
            reinterpret_cast<device mlx_atomic<float>*>(dk_ptr),
            static_cast<float>(dk_val),
            j);
      }

      // Atomic add to dV
      // dV = prob * dO
      for (int j = 0; j < v_per_thread; j++) {
        mlx_atomic_fetch_add_explicit(
            reinterpret_cast<device mlx_atomic<float>*>(dv_ptr),
            static_cast<float>(prob * d_o[j]),
            j);
      }
    }

    // Advance pointers
    k_ptr += inner_k_stride;
    v_ptr += inner_v_stride;
    dk_ptr += inner_k_stride;
    dv_ptr += inner_v_stride;
    mask_offset += BN * mask_kv_seq_stride;
  }

  // Reduce and write dQ
  // NOTE: Use D (head dimension) not BD (simdgroup width) for the stride
  // Each simdgroup needs D elements to store its full dQ contribution
  for (int i = 0; i < qk_per_thread; i++) {
    shared_dQ[simd_gid * D + simd_lid * qk_per_thread + i] = dq[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    for (int i = 0; i < qk_per_thread; i++) {
      U sum_dq = 0;
      for (int sg = 0; sg < BN; sg++) {
        sum_dq += shared_dQ[sg * D + simd_lid * qk_per_thread + i];
      }
      dq_ptr[i] = static_cast<T>(sum_dq);
    }
  }
}
