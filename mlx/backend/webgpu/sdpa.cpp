// Copyright 2026 Apple Inc.
//
// WebGPU fused ScaledDotProductAttention — VECTOR path only (Tq == 1, i.e.
// decode step). The prompt path (Tq > 1) is left on the decomposed fallback.
//
// Algorithm (single-pass chunked FlashAttention-style):
//   For each (batch, query_head), one workgroup of WG threads:
//
//     Phase A — parallel reduction over L (sequence length):
//         m = max_l (score_l)          where score_l = dot(Q, K[l]) * scale (+ mask[l])
//
//     Phase B — process L in chunks of WG threads. For each chunk:
//         Step 1: each thread computes weight_l = exp(score_l - m) for one l,
//                 writes it to shared_red[tid], and accumulates thread_sum.
//         Step 2: threads with tid < D read shared_red and accumulate
//                 o_acc[tid] = sum over l in chunk of (weight_l * V[l, tid]).
//
//     Final: reduce thread_sum to row_sum, normalize O[d] = o_acc[d] / row_sum.
//
// This eliminates the redundant K reads from the old two-phase variant. The
// old Phase B recomputed scores (D threads × L iterations × D K reads =
// L·D² K reads per workgroup); the single-pass variant caches weights in
// shared memory and each K column is read exactly twice (once in Phase A,
// once in Phase B weight compute), dropping K traffic to 2·L·D.
//
// - Workgroup size: 128 threads. For D ≤ 128 each thread owns one output
//   lane; for D = 256 each thread owns D_PER_THREAD = 2 lanes via a
//   strided loop (tid, tid + 128).
// - One workgroup per (batch * n_heads). Dispatch grid = (B*H, 1, 1).
// - Head dim D is baked into the WGSL source as a `const`, so loops over D
//   unroll and the pipeline cache keys by (dtype, D, has_mask, subgroups).
// - bf16 arrays arrive on the GPU already promoted to f32 (see
//   device.cpp::upload_with_conversion), so we emit f32 WGSL types in both
//   the float32 and bfloat16 paths (via dtype_to_wgsl_safe).
// - GQA: kv_head = h / gqa_factor, gqa_factor = H / H_kv. Both K and V are
//   indexed with kv_head, Q with h.
// - Mask (optional): when `has_arr_mask` is true and the mask is a non-bool
//   array, it's broadcast by `fast.cpp` to the Q output shape [B, H, 1, L].
//   We require it to be row-contiguous and read `mask[(b*H + h)*L + l]`
//   directly. Bool masks stay on the fallback (supports_bool_mask = false,
//   so fast.cpp converts them to additive masks before we get here — but we
//   still return true from use_fallback for safety if anything non-standard
//   sneaks through).
//
// Dispatch layout:
//   group(0) binding(0)  : q           [B, H,     1, D]  (read)
//   group(0) binding(1)  : k           [B, H_kv,  L, D]  (read)
//   group(0) binding(2)  : v           [B, H_kv,  L, D]  (read)
//   group(0) binding(3)  : mask        [B, H,     1, L]  (read, optional)
//   group(0) binding(N)  : output      [B, H,     1, D]  (read_write)
//   group(0) binding(N+1): params      (uniform)
//
// Pipeline cache key: "sdpa_v_" + dtype + "_D" + D + (has_mask ? "_m" : "")
//                     + (has_subgroups ? "_sg" : "").

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/fast_primitives.h"

#include <cassert>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>

#ifdef MLX_WGPU_LOG_KERNELS
#include <iostream>
#endif

namespace mlx::core::fast {

namespace {

// SDPA workgroup size: 128 threads. For D ≤ 128, each thread owns at most
// one output lane (threads with tid >= D idle during the per-lane Phase B
// write but still participate in the Phase A reduction). For D = 256, each
// thread owns D_PER_THREAD = 2 output lanes via a strided loop.
constexpr uint32_t SDPA_WORKGROUP_SIZE = 128;

// Uniform struct laid out to match the WGSL definition below.
// Total size = 32 bytes (8 u32/f32 slots). Padded to 256 by the WGPU pool.
struct SdpaParams {
  uint32_t B;          // batch
  uint32_t H;          // query heads
  uint32_t H_kv;       // kv heads
  uint32_t L;          // kv sequence length
  uint32_t D;          // head dim (also baked as a WGSL const)
  uint32_t has_mask;   // 0 or 1
  uint32_t gqa_factor; // H / H_kv
  float    scale;      // scale factor (usually 1/sqrt(D), but caller decides)
};

// ---------------------------------------------------------------------------
// WGSL kernel generation
// ---------------------------------------------------------------------------

std::string make_sdpa_vector_kernel(
    const std::string& entry_name,
    const std::string& dtype,
    uint32_t D,
    bool has_mask,
    bool use_subgroups) {
  std::ostringstream s;

  if (use_subgroups) {
    s << "enable subgroups;\n";
  }
  if (dtype == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  // D_PER_THREAD: number of output lanes per thread. 1 for D ≤ 128, 2 for
  // D = 256. Allows the 128-thread workgroup to cover all D output lanes.
  uint32_t d_per_thread = (D + SDPA_WORKGROUP_SIZE - 1) / SDPA_WORKGROUP_SIZE;

  s << "const WORKGROUP_SIZE: u32 = " << SDPA_WORKGROUP_SIZE << "u;\n";
  s << "const D: u32 = " << D << "u;\n";
  s << "const D_PER_THREAD: u32 = " << d_per_thread << "u;\n\n";

  s << "struct SdpaParams {\n"
    << "  B: u32,\n"
    << "  H: u32,\n"
    << "  H_kv: u32,\n"
    << "  L: u32,\n"
    << "  D: u32,\n"
    << "  has_mask: u32,\n"
    << "  gqa_factor: u32,\n"
    << "  scale: f32,\n"
    << "}\n\n";

  // Bindings: q, k, v, [mask], output, uniform
  uint32_t binding = 0;
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> q: array<"
    << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> k: array<"
    << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> v: array<"
    << dtype << ">;\n";
  if (has_mask) {
    s << "@group(0) @binding(" << binding++
      << ") var<storage, read> mask: array<" << dtype << ">;\n";
  }
  s << "@group(0) @binding(" << binding++
    << ") var<storage, read_write> output: array<" << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++
    << ") var<uniform> params: SdpaParams;\n\n";

  // Workgroup shared memory:
  //   - shared_red  : scratch for Phase A reductions (max then sum, reused)
  //   - shared_scale: Phase A outputs (m, s) broadcast to all threads
  //   - For Phase B we stream scores through shared memory too, reusing
  //     shared_red as "shared_exp" so Phase B doesn't need to recompute scores.
  //     BUT: shared_red has only WORKGROUP_SIZE slots (128), and L can be
  //     much larger, so this optimisation doesn't apply generally. Instead
  //     Phase B recomputes dot(Q, K[l]) — 2x K traffic, but simple.
  //   - shared_q    : Q vector cached once at workgroup start (D floats).
  s << "var<workgroup> shared_red: array<f32, WORKGROUP_SIZE>;\n";
  s << "var<workgroup> shared_m: f32;\n";
  s << "var<workgroup> shared_inv_s: f32;\n";
  s << "var<workgroup> shared_q: array<f32, D>;\n\n";

  // Helper ops for the shared-memory tree reduction.
  s << "fn max_op(a: f32, b: f32) -> f32 { return max(a, b); }\n";
  s << "fn sum_op(a: f32, b: f32) -> f32 { return a + b; }\n\n";

  // Main entry.
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let tid = lid.x;\n"
    << "  let wg = wg_id.x;\n"
    << "  let H = params.H;\n"
    << "  let H_kv = params.H_kv;\n"
    << "  let L = params.L;\n"
    << "  let gqa = params.gqa_factor;\n"
    << "  let scale = params.scale;\n"
    << "  let b = wg / H;\n"
    << "  let h = wg % H;\n"
    << "  let h_kv = h / gqa;\n"
    << "\n"
    << "  // Offsets (element-indexed) for this (b, h) slice:\n"
    << "  //   Q[b, h, 0, :]   = q[b * H  * D + h    * D + ...]     (Tq=1)\n"
    << "  //   K[b, hk, l, :]  = k[(b*H_kv + hk) * L * D + l*D + d]\n"
    << "  //   V[b, hk, l, :]  = v[(b*H_kv + hk) * L * D + l*D + d]\n"
    << "  //   O[b, h, 0, :]   = o[(b*H + h) * D + d]\n"
    << "  let q_base = (b * H + h) * D;\n"
    << "  let kv_base = (b * H_kv + h_kv) * L * D;\n"
    << "  let o_base = (b * H + h) * D;\n";
  if (has_mask) {
    // Mask is broadcast to output query shape [B, H, 1, L]; row-contiguous.
    s << "  let mask_base = (b * H + h) * L;\n";
  }
  s << "\n";

  // Phase 0: cache Q into shared memory. When D > WORKGROUP_SIZE, each
  // thread loads D_PER_THREAD elements at stride WORKGROUP_SIZE.
  s << "  // Phase 0: cooperatively cache Q[b, h, 0, :] into shared memory.\n"
    << "  for (var dd: u32 = 0u; dd < D_PER_THREAD; dd = dd + 1u) {\n"
    << "    let d = tid + dd * WORKGROUP_SIZE;\n"
    << "    if (d < D) {\n"
    << "      shared_q[d] = f32(q[q_base + d]);\n"
    << "    }\n"
    << "  }\n"
    << "  workgroupBarrier();\n"
    << "\n";

  // Phase A-1: compute max of scores over L.
  s << "  // Phase A-1: compute per-thread partial max over L (stride WG).\n"
    << "  var thread_max: f32 = f32(-3.402823e+38);\n"
    << "  for (var l: u32 = tid; l < L; l = l + WORKGROUP_SIZE) {\n"
    << "    var acc: f32 = 0.0;\n"
    << "    let k_row = kv_base + l * D;\n"
    << "    for (var d: u32 = 0u; d < D; d = d + 1u) {\n"
    << "      acc = acc + shared_q[d] * f32(k[k_row + d]);\n"
    << "    }\n"
    << "    var score = acc * scale;\n";
  if (has_mask) {
    s << "    score = score + f32(mask[mask_base + l]);\n";
  }
  s << "    thread_max = max(thread_max, score);\n"
    << "  }\n"
    << "\n";

  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s,
        "thread_max",
        "shared_red",
        "subgroupMax",
        "max_op",
        SDPA_WORKGROUP_SIZE,
        32,
        "mx");
  } else {
    s << "  shared_red[tid] = thread_max;\n"
      << "  workgroupBarrier();\n";
    wgpu::emit_unrolled_reduction(
        s, "shared_red", "max_op", SDPA_WORKGROUP_SIZE);
  }

  // Broadcast the row max via shared memory.
  s << "  if (tid == 0u) { shared_m = shared_red[0]; }\n"
    << "  workgroupBarrier();\n"
    << "  let row_max = shared_m;\n"
    << "\n";

  // Phase B: single-pass chunked weight compute + V accumulation.
  //
  // Process L in chunks of WORKGROUP_SIZE. For each chunk:
  //   1. All threads compute weight_l = exp(score_l - row_max) for
  //      l = chunk_start + tid (if in range); otherwise weight = 0.
  //      Cache weights in shared_red[tid] AND add to thread-local
  //      partial sum (thread_sum).
  //   2. Each thread accumulates D_PER_THREAD output lanes from V,
  //      reading shared_red for cached weights.
  //
  // After the loop, reduce thread_sum across the workgroup to get row_sum,
  // then normalize and write O[d] = o_acc[d] / row_sum. Each thread owns
  // D_PER_THREAD accumulators for lanes {tid, tid+WG, ..., tid+(DPT-1)*WG}.
  s << "  // Phase B: single-pass chunked weight cache + V accumulation.\n"
    << "  var o_acc: array<f32, D_PER_THREAD>;\n"
    << "  for (var dd: u32 = 0u; dd < D_PER_THREAD; dd = dd + 1u) {\n"
    << "    o_acc[dd] = 0.0;\n"
    << "  }\n"
    << "  var thread_sum: f32 = 0.0;\n"
    << "  for (var chunk_start: u32 = 0u; chunk_start < L;\n"
    << "       chunk_start = chunk_start + WORKGROUP_SIZE) {\n"
    << "    let l = chunk_start + tid;\n"
    << "    var weight: f32 = 0.0;\n"
    << "    if (l < L) {\n"
    << "      var acc: f32 = 0.0;\n"
    << "      let k_row = kv_base + l * D;\n"
    << "      for (var d: u32 = 0u; d < D; d = d + 1u) {\n"
    << "        acc = acc + shared_q[d] * f32(k[k_row + d]);\n"
    << "      }\n"
    << "      var score = acc * scale;\n";
  if (has_mask) {
    s << "      score = score + f32(mask[mask_base + l]);\n";
  }
  s << "      weight = exp(score - row_max);\n"
    << "    }\n"
    << "    shared_red[tid] = weight;\n"
    << "    thread_sum = thread_sum + weight;\n"
    << "    workgroupBarrier();\n"
    << "\n"
    << "    // Step 2: V accumulation for D_PER_THREAD output lanes per thread.\n";
  if (d_per_thread == 1) {
    // D <= WG_SIZE: each thread owns at most one lane. Hoist the guard
    // outside the l_inner loop so threads with tid >= D skip entirely.
    s << "    if (tid < D) {\n"
      << "      var chunk_end: u32 = chunk_start + WORKGROUP_SIZE;\n"
      << "      if (chunk_end > L) { chunk_end = L; }\n"
      << "      for (var l_inner: u32 = chunk_start; l_inner < chunk_end;\n"
      << "           l_inner = l_inner + 1u) {\n"
      << "        let w_val = shared_red[l_inner - chunk_start];\n"
      << "        o_acc[0] = o_acc[0] + w_val * f32(v[kv_base + l_inner * D + tid]);\n"
      << "      }\n"
      << "    }\n";
  } else {
    // D > WG_SIZE: each thread owns multiple lanes. Guard inside the dd loop.
    s << "    var chunk_end: u32 = chunk_start + WORKGROUP_SIZE;\n"
      << "    if (chunk_end > L) { chunk_end = L; }\n"
      << "    for (var l_inner: u32 = chunk_start; l_inner < chunk_end;\n"
      << "         l_inner = l_inner + 1u) {\n"
      << "      let w_val = shared_red[l_inner - chunk_start];\n"
      << "      for (var dd: u32 = 0u; dd < D_PER_THREAD; dd = dd + 1u) {\n"
      << "        let d = tid + dd * WORKGROUP_SIZE;\n"
      << "        if (d < D) {\n"
      << "          o_acc[dd] = o_acc[dd] + w_val * f32(v[kv_base + l_inner * D + d]);\n"
      << "        }\n"
      << "      }\n"
      << "    }\n";
  }
  s
    << "    workgroupBarrier();\n"
    << "  }\n"
    << "\n";

  // Reduce thread_sum across the workgroup -> row_sum.
  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s,
        "thread_sum",
        "shared_red",
        "subgroupAdd",
        "sum_op",
        SDPA_WORKGROUP_SIZE,
        32,
        "sm");
  } else {
    s << "  shared_red[tid] = thread_sum;\n"
      << "  workgroupBarrier();\n";
    wgpu::emit_unrolled_reduction(
        s, "shared_red", "sum_op", SDPA_WORKGROUP_SIZE);
  }

  // Broadcast 1/row_sum via shared memory.
  s << "  if (tid == 0u) { shared_inv_s = 1.0 / shared_red[0]; }\n"
    << "  workgroupBarrier();\n"
    << "  let inv_s = shared_inv_s;\n"
    << "\n";

  // Normalize and write the output. Each thread writes D_PER_THREAD lanes.
  s << "  for (var dd: u32 = 0u; dd < D_PER_THREAD; dd = dd + 1u) {\n"
    << "    let d = tid + dd * WORKGROUP_SIZE;\n"
    << "    if (d < D) {\n"
    << "      output[o_base + d] = " << dtype << "(o_acc[dd] * inv_s);\n"
    << "    }\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Tile (prefill) SDPA params + kernel generator.
//
// Handles the Tq > 1 prompt path: one workgroup per (batch, q_head, q_tile),
// each workgroup processing BQ rows of Q × the entire K/V sequence in BK-col
// chunks using FlashAttention-2 online softmax. The result is a single fused
// kernel that avoids the HBM round-trip of the decomposed matmul→softmax→
// matmul fallback.
//
// Tile shape (fixed at BQ=16, BK=8, 128 threads = BQ * BK):
//   Workgroup layout  : (BK, BQ, 1) = (8, 16, 1), local_id.x = tk, .y = tq.
//   Q_smem            : array<f32, BQ * D>        (~16 KiB at D=256)
//   KV_smem           : array<f32, BK * D>        (~8 KiB  at D=256)
//   S_smem            : array<f32, BQ * BK>       (~0.5 KiB)
//   m_i / l_i         : array<f32, BQ>            (per-row running max/sum)
//   rescale           : array<f32, BQ>            (exp(m_old - m_new) per row)
//
// Total shared memory at D=256: ~25 KiB. Fits Chromium's 32 KiB default
// maxComputeWorkgroupStorageSize. For devices with the WebGPU-minimum 16 KiB
// budget we reject Tq > 1 in use_fallback() at D=256; D ≤ 128 still fits.
//
// The KV_smem buffer is re-used across the K load and the V load within one
// KV-block iteration: we load K, compute S/P, then overwrite KV_smem with V
// and do the V · P accumulation. P values live in S_smem across both halves.
//
// Each thread owns one (tq, tk) pair for the S computation and a strided
// slice of the output: O[tq, d] for d ∈ {tk, tk+BK, ..., tk+(D/BK - 1)*BK}.
// O accumulators live in thread-local registers (d_per_thread = D / BK).
//
// Subgroups are not used here — the per-row reductions span only BK=8 values
// and run as a scalar loop inside each thread. Every thread in the row
// computes the same row_max / row_sum independently from S_smem, so there's
// no cross-lane communication to accelerate. (The BK=8 scalar loop turns out
// to be cheaper than a subgroup dance anyway; run-34 footgun avoided.)

constexpr uint32_t SDPA_TILE_BQ = 16;
constexpr uint32_t SDPA_TILE_BK = 8;
constexpr uint32_t SDPA_TILE_WG_SIZE = SDPA_TILE_BQ * SDPA_TILE_BK; // 128

// Uniform struct laid out to match the WGSL struct below. 48 bytes.
struct SdpaTileParams {
  uint32_t B;          // batch
  uint32_t H;          // query heads
  uint32_t H_kv;       // kv heads
  uint32_t L;          // kv sequence length
  uint32_t Tq;         // query sequence length (> 1)
  uint32_t D;          // head dim (also baked as WGSL const)
  uint32_t has_mask;   // 0 or 1
  uint32_t do_causal;  // 0 or 1
  uint32_t gqa_factor; // H / H_kv
  uint32_t q_offset;   // causal offset (== L - Tq for standard causal decode)
  float    scale;      // attention scale (typically 1/sqrt(D))
  uint32_t pad;        // 16-byte alignment
};

std::string make_sdpa_tile_kernel(
    const std::string& entry_name,
    const std::string& dtype,
    uint32_t D,
    bool has_mask,
    bool do_causal) {
  std::ostringstream s;

  if (dtype == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  const uint32_t BQ = SDPA_TILE_BQ;
  const uint32_t BK = SDPA_TILE_BK;
  const uint32_t WG = SDPA_TILE_WG_SIZE;
  // d_per_thread = D / BK. D is always a multiple of BK (BK=8 and D in
  // {64, 96, 128, 256}) — enforced by use_fallback().
  const uint32_t DPT = D / BK;

  s << "const D: u32 = " << D << "u;\n"
    << "const BQ: u32 = " << BQ << "u;\n"
    << "const BK: u32 = " << BK << "u;\n"
    << "const WG: u32 = " << WG << "u;\n"
    << "const DPT: u32 = " << DPT << "u;\n"
    << "const NEG_INF: f32 = -3.402823e+38;\n\n";

  s << "struct SdpaTileParams {\n"
    << "  B: u32,\n"
    << "  H: u32,\n"
    << "  H_kv: u32,\n"
    << "  L: u32,\n"
    << "  Tq: u32,\n"
    << "  D: u32,\n"
    << "  has_mask: u32,\n"
    << "  do_causal: u32,\n"
    << "  gqa_factor: u32,\n"
    << "  q_offset: u32,\n"
    << "  scale: f32,\n"
    << "  pad: u32,\n"
    << "}\n\n";

  // Bindings: q, k, v, [mask], output, uniform
  uint32_t binding = 0;
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> q: array<"
    << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> k: array<"
    << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> v: array<"
    << dtype << ">;\n";
  if (has_mask) {
    s << "@group(0) @binding(" << binding++
      << ") var<storage, read> mask: array<" << dtype << ">;\n";
  }
  s << "@group(0) @binding(" << binding++
    << ") var<storage, read_write> output: array<" << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++
    << ") var<uniform> params: SdpaTileParams;\n\n";

  // Shared memory: Q tile, K/V tile (ping-pong within a block), S tile,
  // per-row running state.
  s << "var<workgroup> Q_smem: array<f32, BQ * D>;\n";
  s << "var<workgroup> KV_smem: array<f32, BK * D>;\n";
  s << "var<workgroup> S_smem: array<f32, BQ * BK>;\n";
  s << "var<workgroup> m_i: array<f32, BQ>;\n";
  s << "var<workgroup> l_i: array<f32, BQ>;\n";
  s << "var<workgroup> rescale_smem: array<f32, BQ>;\n\n";

  // Entry point. Workgroup = (BK, BQ, 1), so local_id.x is the column lane
  // (tk) and local_id.y is the row lane (tq). Dispatch grid: (qTiles, H, B).
  s << "@compute @workgroup_size(" << BK << ", " << BQ << ", 1)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let tk = lid.x;\n"
    << "  let tq = lid.y;\n"
    << "  let flat = tq * BK + tk;  // 0..WG-1\n"
    << "  let q_tile = wg_id.x;     // 0..ceil(Tq/BQ)-1\n"
    << "  let h = wg_id.y;          // 0..H-1\n"
    << "  let b = wg_id.z;          // 0..B-1\n"
    << "  let H_kv = params.H_kv;\n"
    << "  let H = params.H;\n"
    << "  let gqa = params.gqa_factor;\n"
    << "  let h_kv = h / gqa;\n"
    << "  let L = params.L;\n"
    << "  let Tq = params.Tq;\n"
    << "  let scale = params.scale;\n"
    << "  let q_row_base = q_tile * BQ;\n"
    << "  let q_row_abs_base = q_row_base + params.q_offset;  // for causal\n"
    << "\n"
    << "  // Element-indexed offsets into flat row-contiguous Q/K/V/O:\n"
    << "  //   Q[b, h, tq_row, :]   = q[(b*H    + h   )*Tq*D + tq_row*D + d]\n"
    << "  //   K[b, h_kv, k_row, :] = k[(b*H_kv + h_kv)*L *D + k_row*D  + d]\n"
    << "  //   V[b, h_kv, k_row, :] = v[(b*H_kv + h_kv)*L *D + k_row*D  + d]\n"
    << "  //   O[b, h, tq_row, :]   = output[(b*H + h)*Tq*D + tq_row*D + d]\n"
    << "  let q_base = (b * H + h) * Tq * D;\n"
    << "  let kv_base = (b * H_kv + h_kv) * L * D;\n"
    << "  let o_base = q_base;\n";
  if (has_mask) {
    // Mask is broadcast by fast.cpp to [B, H, Tq, L] and made contiguous.
    s << "  let mask_base = ((b * H + h) * Tq) * L;\n";
  }
  s << "\n";

  // Phase 0: cooperatively cache Q[q_tile, :] into shared memory.
  // Each thread loads Q_smem[q_row*D + d] for a strided set of (q_row, d).
  // Out-of-range q_rows (tail tile) get zeroed — the final store guards on
  // q_row < Tq so the bogus zero-Q contributions never leak into output.
  s << "  // Phase 0: cooperatively load Q[q_tile*BQ .. q_tile*BQ+BQ, :] into\n"
    << "  // Q_smem. Each thread loads (BQ*D)/WG = D*BQ/WG = D/BK elements.\n"
    << "  for (var i: u32 = flat; i < BQ * D; i = i + WG) {\n"
    << "    let row = i / D;\n"
    << "    let col = i % D;\n"
    << "    let q_row = q_row_base + row;\n"
    << "    var qv: f32 = 0.0;\n"
    << "    if (q_row < Tq) {\n"
    << "      qv = f32(q[q_base + q_row * D + col]);\n"
    << "    }\n"
    << "    Q_smem[row * D + col] = qv;\n"
    << "  }\n"
    << "\n";

  // Per-row init: m_i = -inf, l_i = 0.
  s << "  // Initialize per-row softmax state.\n"
    << "  if (flat < BQ) {\n"
    << "    m_i[flat] = NEG_INF;\n"
    << "    l_i[flat] = 0.0;\n"
    << "  }\n"
    << "\n"
    << "  // Per-thread register accumulator: O_accum[d_slab] where\n"
    << "  // d_slab ∈ {tk, tk+BK, ..., tk+(DPT-1)*BK}. All initialized to 0.\n"
    << "  var o_acc: array<f32, DPT>;\n"
    << "  for (var i: u32 = 0u; i < DPT; i = i + 1u) { o_acc[i] = 0.0; }\n"
    << "\n"
    << "  workgroupBarrier();\n"
    << "\n";

  // Main loop over KV blocks.
  s << "  // Loop over KV blocks of size BK.\n"
    << "  let num_kv_blocks = (L + BK - 1u) / BK;\n"
    << "  for (var kb: u32 = 0u; kb < num_kv_blocks; kb = kb + 1u) {\n"
    << "    let k_block_start = kb * BK;\n"
    << "\n"
    << "    // Cooperatively load K[k_block_start .. +BK, :] into KV_smem.\n"
    << "    // (BK*D)/WG elements per thread = D/BK = DPT elements.\n"
    << "    for (var i: u32 = flat; i < BK * D; i = i + WG) {\n"
    << "      let row = i / D;\n"
    << "      let col = i % D;\n"
    << "      let k_row = k_block_start + row;\n"
    << "      var kv: f32 = 0.0;\n"
    << "      if (k_row < L) {\n"
    << "        kv = f32(k[kv_base + k_row * D + col]);\n"
    << "      }\n"
    << "      KV_smem[row * D + col] = kv;\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "\n";

  // Step 2: compute S[tq, tk] = dot(Q_smem[tq, :], KV_smem[tk, :]) * scale.
  // Each thread computes exactly one S value. Apply masks.
  s << "    // S[tq, tk] = Q_smem[tq, :] dot KV_smem[tk, :] * scale.\n"
    << "    var acc: f32 = 0.0;\n"
    << "    for (var d: u32 = 0u; d < D; d = d + 1u) {\n"
    << "      acc = acc + Q_smem[tq * D + d] * KV_smem[tk * D + d];\n"
    << "    }\n"
    << "    var s_val = acc * scale;\n"
    << "\n"
    << "    // Per-element masks for (q_row, k_col).\n"
    << "    let q_row = q_row_base + tq;\n"
    << "    let k_col = k_block_start + tk;\n"
    << "    let q_row_abs = q_row_abs_base + tq;\n";
  if (do_causal) {
    s << "    if (k_col > q_row_abs) { s_val = NEG_INF; }\n";
  }
  s << "    if (k_col >= L) { s_val = NEG_INF; }\n"
    << "    if (q_row >= Tq) { s_val = NEG_INF; }\n";
  if (has_mask) {
    // Mask is [B, H, Tq, L] row-contiguous; bounds already checked by the
    // k_col/q_row guards above — safe to index directly.
    s << "    if (q_row < Tq && k_col < L) {\n"
      << "      s_val = s_val + f32(mask[mask_base + q_row * L + k_col]);\n"
      << "    }\n";
  }
  s << "    S_smem[tq * BK + tk] = s_val;\n"
    << "    workgroupBarrier();\n"
    << "\n";

  // Step 3: per-row online softmax update.
  // Each thread scans its own row once and computes row_max locally.
  // The first lane of each row (tk == 0) updates the shared m_i/l_i/rescale.
  // All threads then read the shared rescale to update their O_accum
  // registers and read the updated m_i to compute P = exp(S - m_new).
  s << "    // Per-row scalar reduction over BK=" << BK << " S values.\n"
    << "    // All threads in a row compute the same row_max independently.\n"
    << "    var row_max_block: f32 = NEG_INF;\n"
    << "    for (var j: u32 = 0u; j < BK; j = j + 1u) {\n"
    << "      row_max_block = max(row_max_block, S_smem[tq * BK + j]);\n"
    << "    }\n"
    << "    // FlashAttention online softmax rescale. Row lead (tk == 0)\n"
    << "    // writes the shared running state so every thread in the row\n"
    << "    // can read a consistent rescale factor and m_new below.\n"
    << "    if (tk == 0u) {\n"
    << "      let m_old = m_i[tq];\n"
    << "      let m_new = max(m_old, row_max_block);\n"
    << "      let rescale = exp(m_old - m_new);\n"
    << "      // Compute new l_i from old l_i · rescale + sum(exp(S - m_new))\n"
    << "      // of THIS block. The running l_i rescale happens here; the\n"
    << "      // O_accum rescale is applied per-thread below.\n"
    << "      var row_sum_block: f32 = 0.0;\n"
    << "      for (var j: u32 = 0u; j < BK; j = j + 1u) {\n"
    << "        row_sum_block = row_sum_block + exp(S_smem[tq * BK + j] - m_new);\n"
    << "      }\n"
    << "      m_i[tq] = m_new;\n"
    << "      l_i[tq] = l_i[tq] * rescale + row_sum_block;\n"
    << "      rescale_smem[tq] = rescale;\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "\n"
    << "    // All threads: rescale O_accum registers by the row's factor,\n"
    << "    // then replace S with P = exp(S - m_new) in shared memory.\n"
    << "    let m_new = m_i[tq];\n"
    << "    let rescale = rescale_smem[tq];\n"
    << "    for (var i: u32 = 0u; i < DPT; i = i + 1u) {\n"
    << "      o_acc[i] = o_acc[i] * rescale;\n"
    << "    }\n"
    << "    S_smem[tq * BK + tk] = exp(S_smem[tq * BK + tk] - m_new);\n"
    << "    workgroupBarrier();\n"
    << "\n";

  // Step 4: load V into KV_smem (reusing the K slot), then accumulate
  // O_accum[tq, d] += sum_k P[tq, k] * V_smem[k, d] across the owned d slab.
  s << "    // Load V[k_block_start .. +BK, :] into KV_smem (overwrites K).\n"
    << "    for (var i: u32 = flat; i < BK * D; i = i + WG) {\n"
    << "      let row = i / D;\n"
    << "      let col = i % D;\n"
    << "      let k_row = k_block_start + row;\n"
    << "      var vv: f32 = 0.0;\n"
    << "      if (k_row < L) {\n"
    << "        vv = f32(v[kv_base + k_row * D + col]);\n"
    << "      }\n"
    << "      KV_smem[row * D + col] = vv;\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "\n"
    << "    // O_accum[tq, d] += sum_k P[tq, k] * V_smem[k, d]\n"
    << "    // Thread (tq, tk) owns d_slab = {tk, tk+BK, ..., tk+(DPT-1)*BK}.\n"
    << "    for (var di: u32 = 0u; di < DPT; di = di + 1u) {\n"
    << "      let d = di * BK + tk;\n"
    << "      var acc_d: f32 = o_acc[di];\n"
    << "      for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {\n"
    << "        let p = S_smem[tq * BK + kk];\n"
    << "        acc_d = acc_d + p * KV_smem[kk * D + d];\n"
    << "      }\n"
    << "      o_acc[di] = acc_d;\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "  }\n"
    << "\n";

  // Final normalize + store. Guard on q_row < Tq for tail tile correctness.
  s << "  // Normalize by l_i[tq] and write O[q_row, d] for owned d slab.\n"
    << "  let q_row = q_row_base + tq;\n"
    << "  if (q_row < Tq) {\n"
    << "    let inv_l = 1.0 / l_i[tq];\n"
    << "    for (var di: u32 = 0u; di < DPT; di = di + 1u) {\n"
    << "      let d = di * BK + tk;\n"
    << "      output[o_base + q_row * D + d] = " << dtype
    << "(o_acc[di] * inv_l);\n"
    << "    }\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// ScaledDotProductAttention::use_fallback
// ---------------------------------------------------------------------------

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  // Runtime kill switch from ?sdpa_fallback=1 (demo A/B). Forces the
  // decomposed matmul→softmax→matmul path unconditionally.
  if (wgpu::sdpa_fallback_forced()) {
    return true;
  }
  // Training / logsumexp path uses the VJP-friendly decomposed kernel.
  if (is_training || output_logsumexp) {
    return true;
  }
  // Sanity: Q/K/V must have at least [B, H, T, D] layout.
  if (q.ndim() < 2 || k.ndim() < 2 || v.ndim() < 2) {
    return true;
  }

  // Head dim restriction. Both the vector path (Tq==1) and tile path (Tq>1)
  // support D in {64, 96, 128, 256}. At D=256, the vector kernel uses
  // D_PER_THREAD=2 so each of the 128 threads owns 2 output lanes.
  // D must also be divisible by SDPA_TILE_BK=8 so the tile kernel's
  // d_per_thread slab math works.
  int d = q.shape(-1);
  if (d != 64 && d != 96 && d != 128 && d != 256) {
    return true;
  }
  if (k.shape(-1) != d || v.shape(-1) != d) {
    return true;
  }

  // K and V must have matching sequence length.
  if (k.shape(-2) != v.shape(-2)) {
    return true;
  }

  // Dtype restriction: float32 or bfloat16 (which is promoted to f32 on GPU).
  auto dt = q.dtype();
  if (dt != float32 && dt != bfloat16) {
    return true;
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    return true;
  }

  // Tq == 1  -> vector kernel path (decode, D_PER_THREAD strided loop).
  // Tq  > 1  -> tile kernel path (prefill).
  // Both paths support D == 256. The tile path at D == 256 needs ~25 KiB
  // shared memory (fits Chromium's 32 KiB); use ?sdpa_fallback=1 on tighter
  // devices. The vector path at D == 256 uses only D*4 = 1 KiB for shared_q
  // plus 512 bytes for shared_red — well within any budget.

  // Mask handling:
  //   - No mask -> OK.
  //   - Causal (has_mask && do_causal && !has_arr_mask) -> OK: the tile
  //     kernel implements causal masking internally via q_offset. The
  //     vector kernel handles do_causal via the SdpaParams.has_mask flag.
  //   - Array mask (non-bool) -> OK (we read it as additive float).
  //   - Array mask (bool) -> NOT OK: we say supports_bool_mask() == false,
  //     so fast.cpp will convert it to an additive float mask before reaching
  //     eval_gpu. We still accept has_arr_mask here.
  //   - Scalar bool mask (has_mask && !has_arr_mask && !do_causal) -> fall back.
  if (has_mask && !has_arr_mask && !do_causal) {
    return true;
  }

  // Row-contiguity: the kernel indexes Q/K/V as flat arrays. Non-contig
  // inputs (e.g. from transpose views) are made contiguous by eval_gpu's
  // ensure_contig helper, so we do NOT reject them here. This is important
  // because the common Qwen3.5 prefill path transposes Q/K/V from
  // [B, T, H, D] to [B, H, T, D], producing non-contiguous views.
  //
  // NOTE on mask row-contiguity: fast.cpp broadcasts the incoming mask to
  // shape [B, H, Tq, L], which often produces strided (broadcast) arrays.
  // The kernel assumes row-contiguous mask layout for that shape and
  // indexes it as `mask[((b*H + h)*Tq + q_row) * L + k_col]`. We cannot
  // check the mask array here (it's built after use_fallback runs), so
  // eval_gpu makes the mask contiguous via contiguous_copy_gpu if needed.

  return false;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

// ---------------------------------------------------------------------------
// ScaledDotProductAttention::eval_gpu
// ---------------------------------------------------------------------------

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Inputs layout from fast.cpp:
  //   inputs[0] = Q
  //   inputs[1] = K
  //   inputs[2] = V
  //   inputs[3] = mask (if has_arr_mask; always present when we accept a mask)
  //   inputs[back] = sinks (we do not support sinks; use_fallback rejects that
  //                         path earlier if the sinks flag is set, but defend
  //                         here anyway).
  assert(outputs.size() == 1);
  auto& out = outputs[0];
  auto& s = stream();

  if (has_sinks_) {
    // The vector kernel does not support attention sinks. use_fallback is a
    // static method and does not receive the has_sinks flag, so we can't
    // reject sinks earlier. For Qwen3.5 inference (the only consumer of this
    // path today) sinks are never set, so this throw is strictly defensive.
    throw std::runtime_error(
        "[WebGPU sdpa] attention sinks are not supported by the fused "
        "vector kernel.");
  }

  // Figure out whether we have an array mask. Mirrors the Metal backend's
  // convention: has_arr_mask when inputs has more than (3 + has_sinks_)
  // entries. Since we rejected has_sinks_ above, this reduces to
  // inputs.size() > 3.
  bool has_mask = inputs.size() > 3;
  if (inputs.size() != 3 && inputs.size() != 4) {
    throw std::runtime_error(
        "[WebGPU sdpa] Unexpected input count " +
        std::to_string(inputs.size()) + "; expected 3 or 4.");
  }

  const array& q = inputs[0];
  const array& k = inputs[1];
  const array& v = inputs[2];

  // Shapes: Q = [B, H, Tq, D], K/V = [B, H_kv, L, D].
  uint32_t B = static_cast<uint32_t>(q.shape(0));
  uint32_t H = static_cast<uint32_t>(q.shape(1));
  uint32_t H_kv = static_cast<uint32_t>(k.shape(1));
  uint32_t Tq = static_cast<uint32_t>(q.shape(-2));
  uint32_t L = static_cast<uint32_t>(k.shape(-2));
  uint32_t D = static_cast<uint32_t>(q.shape(-1));

  if (H_kv == 0 || H % H_kv != 0) {
    throw std::runtime_error(
        std::string("[WebGPU sdpa] invalid head counts H=") +
        std::to_string(H) + " H_kv=" + std::to_string(H_kv));
  }
  uint32_t gqa_factor = H / H_kv;

  // Ensure Q/K/V have the right row-contiguous, zero-offset layout. If the
  // upstream arrays are offset (e.g. due to slicing) we fall back to a
  // defensive contiguous copy. use_fallback already rejected non-contig
  // layouts, but offset != 0 can still slip through.
  auto ensure_contig = [&](const array& a) -> array {
    if (a.flags().row_contiguous && a.offset() == 0) {
      return a;
    }
    array c = contiguous_copy_gpu(a, s);
    auto& enc = wgpu::get_command_encoder(s);
    enc.add_temporary(c);
    return c;
  };
  array q_c = ensure_contig(q);
  array k_c = ensure_contig(k);
  array v_c = ensure_contig(v);
  // The mask, if present, is usually a broadcast view of a smaller mask
  // (e.g. [1, 1, 1, L] broadcast to [B, H, 1, L] or [B, H, Tq, L]); make
  // it contiguous so the kernel can read it with a flat offset.
  std::optional<array> mask_c_opt;
  if (has_mask) {
    mask_c_opt = ensure_contig(inputs[3]);
  }

  // Allocate the output.
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  wgpu::ensure_wgpu_size(out);

  if (B == 0 || H == 0 || D == 0) {
    return;
  }
  // L == 0 is undefined for softmax(Q*K^T); zero the output and return.
  if (L == 0) {
    // Leave output as uninitialised allocation — MLX upstream should never
    // hand us L == 0 for a real attention call. Do nothing; safer than
    // dispatching over an empty range.
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  // Choose WGSL dtype. bf16 is uploaded as f32 by device.cpp, so
  // dtype_to_wgsl_safe returns "f32" for bf16 already.
  const char* dtype_wgsl = wgpu::dtype_to_wgsl_safe(q.dtype());
  std::string dtype(dtype_wgsl);

  // Branch on Tq: vector (Tq==1) vs tile (Tq>1).
  if (Tq == 1) {
    // ---------------- Vector kernel path (decode, Tq==1) ----------------
    bool use_sg = dev.has_subgroups();
    std::string sg_suffix = use_sg ? "_sg" : "";
    std::string mask_suffix = has_mask ? "_m" : "";
    std::string entry_name = std::string("sdpa_v_") + dtype + "_D" +
        std::to_string(D) + mask_suffix + sg_suffix;

    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name, [&]() {
#ifdef MLX_WGPU_LOG_KERNELS
          // First-dispatch log — the shader-module cache only invokes this
          // lambda on a MISS, so each unique entry_name prints exactly once.
          std::cerr << "[MLX-KERNEL] " << entry_name << std::endl;
#endif
          return make_sdpa_vector_kernel(
              entry_name, dtype, D, has_mask, use_sg);
        });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    // Register inputs/outputs with the encoder for bookkeeping.
    encoder.set_input_array(q_c);
    encoder.set_input_array(k_c);
    encoder.set_input_array(v_c);
    if (has_mask) {
      encoder.set_input_array(*mask_c_opt);
    }
    encoder.set_output_array(out);

    // Populate uniform buffer.
    SdpaParams params{};
    params.B = B;
    params.H = H;
    params.H_kv = H_kv;
    params.L = L;
    params.D = D;
    params.has_mask = has_mask ? 1u : 0u;
    params.gqa_factor = gqa_factor;
    params.scale = scale_;

    auto& pool = dev.uniform_pool();
    WGPUBuffer uniform_buf =
        pool.acquire(dev.gpu_queue(), &params, sizeof(SdpaParams));

    // Build bind group.
    WGPUBuffer q_buf = wgpu::wgpu_buffer(q_c);
    WGPUBuffer k_buf = wgpu::wgpu_buffer(k_c);
    WGPUBuffer v_buf = wgpu::wgpu_buffer(v_c);
    WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

    std::vector<std::pair<WGPUBuffer, uint64_t>> bg_entries;
    bg_entries.reserve(has_mask ? 6 : 5);
    bg_entries.emplace_back(q_buf, wgpuBufferGetSize(q_buf));
    bg_entries.emplace_back(k_buf, wgpuBufferGetSize(k_buf));
    bg_entries.emplace_back(v_buf, wgpuBufferGetSize(v_buf));
    if (has_mask) {
      WGPUBuffer mask_buf = wgpu::wgpu_buffer(*mask_c_opt);
      bg_entries.emplace_back(mask_buf, wgpuBufferGetSize(mask_buf));
    }
    bg_entries.emplace_back(out_buf, wgpuBufferGetSize(out_buf));
    bg_entries.emplace_back(uniform_buf, sizeof(SdpaParams));

    WGPUBindGroup bg = wgpu::create_bind_group(pe.layout, bg_entries);

    // One workgroup per (batch, head).
    encoder.dispatch_compute(pe.pipeline, bg, B * H, 1u, 1u);

    wgpuBindGroupRelease(bg);
    encoder.add_completed_handler([uniform_buf]() {
      wgpu::device().uniform_pool().release(uniform_buf);
    });
    return;
  }

  // ---------------- Tile kernel path (prefill, Tq>1) ----------------
  // Shared-memory budget: at BQ=16, BK=8 the kernel needs
  //   Q_smem : BQ*D*4  = 64*D  bytes
  //   KV_smem: BK*D*4  = 32*D  bytes
  //   S_smem : BQ*BK*4 = 512   bytes
  //   m_i/l_i/rescale: 3 * BQ * 4 = 192 bytes
  // At D=256 that's 96*256 + 512 + 192 = 25280 bytes (~25 KiB).
  // Chromium's default maxComputeWorkgroupStorageSize on M3 is 32 KiB;
  // WebGPU minimum is 16 KiB. D in {64, 96, 128} fit comfortably in
  // 16 KiB. D=256 needs ~25 KiB, which exceeds the 16 KiB minimum but
  // fits Chromium's 32 KiB — use ?sdpa_fallback=1 on tighter devices.
  //
  // NOTE: We intentionally do NOT query wgpuDeviceGetLimits here. The
  // WGPULimits struct contains uint64_t fields that have 8-byte alignment
  // in WASM32, introducing padding that the JS-side bridge does not
  // account for. This causes maxComputeWorkgroupStorageSize to read as
  // garbage. Since use_fallback already constrains D to {64,96,128,256},
  // all of which fit Chromium's 32 KiB, the static gate is sufficient.

  // Causal offset: Tq query rows are the LAST Tq rows of a length-L sequence,
  // so the first q row's absolute index is (L - Tq). The tile kernel guards
  // `k_col > q_row + q_offset`.
  uint32_t q_offset = (L >= Tq) ? (L - Tq) : 0u;
  bool do_causal = do_causal_;

  std::string mask_suffix = has_mask ? "_m" : "";
  std::string causal_suffix = do_causal ? "_c" : "";
  std::string entry_name = std::string("sdpa_tile_") + dtype + "_D" +
      std::to_string(D) + mask_suffix + causal_suffix;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name, [&]() {
#ifdef MLX_WGPU_LOG_KERNELS
        std::cerr << "[MLX-KERNEL] " << entry_name << std::endl;
#endif
        return make_sdpa_tile_kernel(
            entry_name, dtype, D, has_mask, do_causal);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(q_c);
  encoder.set_input_array(k_c);
  encoder.set_input_array(v_c);
  if (has_mask) {
    encoder.set_input_array(*mask_c_opt);
  }
  encoder.set_output_array(out);

  SdpaTileParams params{};
  params.B = B;
  params.H = H;
  params.H_kv = H_kv;
  params.L = L;
  params.Tq = Tq;
  params.D = D;
  params.has_mask = has_mask ? 1u : 0u;
  params.do_causal = do_causal ? 1u : 0u;
  params.gqa_factor = gqa_factor;
  params.q_offset = q_offset;
  params.scale = scale_;
  params.pad = 0u;

  auto& pool = dev.uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(dev.gpu_queue(), &params, sizeof(SdpaTileParams));

  WGPUBuffer q_buf = wgpu::wgpu_buffer(q_c);
  WGPUBuffer k_buf = wgpu::wgpu_buffer(k_c);
  WGPUBuffer v_buf = wgpu::wgpu_buffer(v_c);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  std::vector<std::pair<WGPUBuffer, uint64_t>> bg_entries;
  bg_entries.reserve(has_mask ? 6 : 5);
  bg_entries.emplace_back(q_buf, wgpuBufferGetSize(q_buf));
  bg_entries.emplace_back(k_buf, wgpuBufferGetSize(k_buf));
  bg_entries.emplace_back(v_buf, wgpuBufferGetSize(v_buf));
  if (has_mask) {
    WGPUBuffer mask_buf = wgpu::wgpu_buffer(*mask_c_opt);
    bg_entries.emplace_back(mask_buf, wgpuBufferGetSize(mask_buf));
  }
  bg_entries.emplace_back(out_buf, wgpuBufferGetSize(out_buf));
  bg_entries.emplace_back(uniform_buf, sizeof(SdpaTileParams));

  WGPUBindGroup bg = wgpu::create_bind_group(pe.layout, bg_entries);

  // Dispatch grid: (qTiles, H, B). qTiles = ceil(Tq / BQ).
  uint32_t qTiles = (Tq + SDPA_TILE_BQ - 1u) / SDPA_TILE_BQ;
  encoder.dispatch_compute(pe.pipeline, bg, qTiles, H, B);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core::fast
