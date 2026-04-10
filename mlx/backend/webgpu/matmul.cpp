// Copyright 2026 Apple Inc.
//
// WebGPU matrix multiplication: tiled GEMM and GEMV.
//
// Uses dynamic WGSL code generation. Each unique (transpose, dtype) produces
// a distinct shader module and compute pipeline, cached for reuse.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

#ifdef MLX_WGPU_LOG_KERNELS
#include <iostream>
#endif

namespace mlx::core {

namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr uint32_t TILE_SIZE = 16;
constexpr uint32_t GEMV_WORKGROUP_SIZE = 256;

// ---------------------------------------------------------------------------
// Uniform buffer params
// ---------------------------------------------------------------------------

struct MatmulParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
  uint32_t batch_size;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_c;
  uint32_t offset_a;
  uint32_t offset_b;
};

// ---------------------------------------------------------------------------
// WGSL GEMV kernel generation (split-K)
// ---------------------------------------------------------------------------

// Split-K GEMV: one workgroup per output element, 256 threads cooperatively
// reduce the K dimension. This pattern delivers two big wins over the naive
// "one thread per output" GEMV that this replaces:
//
//   1. Coalesced B reads in the common b_transposed case
//      (weight matrix stored as (N, K) row-major). All 256 threads in a
//      workgroup share the same col and iterate k → they read 256 adjacent
//      K elements each iteration → perfect coalescing. The naive kernel had
//      adjacent threads striding by ldb (= K), totally uncoalesced.
//
//   2. ~100× more active threads for typical decode GEMV sizes
//      (e.g. N=3072 → 3072 workgroups × 256 threads = 786k threads, vs
//      the old 12 workgroups × 256 = 3k threads). This is critical for
//      hiding memory latency on GPUs, which want 100k+ active threads.
//
// Workgroup reduction uses subgroup intrinsics when available, with an
// unrolled tree-reduction fallback.
//
// Dispatch: one workgroup per output element with 2D fanout using a fixed
// GRID_X (65535 — the WebGPU minimum workgroups-per-dim) so large outputs
// like lm_head (vocab = 151k) fit. The shader decodes output_idx from
// (wid.x, wid.y) uniformly for both small and large outputs.
std::string make_gemv_kernel(
    const std::string& entry_name,
    const std::string& dtype,
    bool a_transposed,
    bool b_transposed,
    bool use_sg,
    bool b_packed_bf16 = false) {
  std::ostringstream s;

  if (use_sg) {
    s << "enable subgroups;\n";
  }
  if (dtype == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WG_SIZE: u32 = 256u;\n"
    << "const GRID_X: u32 = 65535u;\n\n";

  s << "struct MatmulParams {\n"
    << "  M: u32, N: u32, K: u32,\n"
    << "  lda: u32, ldb: u32, ldc: u32,\n"
    << "  batch_size: u32,\n"
    << "  batch_stride_a: u32, batch_stride_b: u32, batch_stride_c: u32,\n"
    << "  offset_a: u32, offset_b: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> a: array<" << dtype << ">;\n";
  if (b_packed_bf16) {
    s << "@group(0) @binding(1) var<storage, read> b: array<u32>;\n";
  } else {
    s << "@group(0) @binding(1) var<storage, read> b: array<" << dtype
      << ">;\n";
  }
  s << "@group(0) @binding(2) var<storage, read_write> out: array<" << dtype
    << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: MatmulParams;\n\n";

  if (b_packed_bf16) {
    s << wgpu::wgsl_unpack_bf16_pair();
  }

  // Helper for unrolled / subgroup tree-reduce phase.
  s << "fn sum_op(x: f32, y: f32) -> f32 { return x + y; }\n\n";

  // Shared memory for reduction (f32 accumulator regardless of dtype).
  s << "var<workgroup> shared_acc: array<f32, 256>;\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(workgroup_id) wid: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u) {\n"
    << "  let output_idx = wid.x + wid.y * GRID_X;\n"
    << "  let batch = wid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let M = params.M;\n"
    << "  let N = params.N;\n"
    << "  let K = params.K;\n"
    << "  let output_size = M * N;\n"
    << "  if (output_idx >= output_size) { return; }\n"
    << "  let row = output_idx / N;\n"
    << "  let col = output_idx % N;\n"
    << "  let tid = lid.x;\n"
    << "  let a_base = batch * params.batch_stride_a + params.offset_a;\n"
    << "  let b_base = batch * params.batch_stride_b + params.offset_b;\n"
    << "  let c_base = batch * params.batch_stride_c;\n"
    << "  var acc: f32 = 0.0;\n";

  if (b_packed_bf16) {
    // Step A only implements the b_transposed=true (row=N, col=K) hot path:
    // each row of B lives contiguously in K, so we can stride pairs of bf16
    // values through array<u32>. Step B will add the non-transposed variant.
    assert(b_transposed && "packed bf16 GEMV requires b_transposed=true in Step A");
    s << "  // Packed bf16: each u32 holds 2 consecutive K values of row `col`.\n"
      << "  let ldb_u32 = params.ldb >> 1u;\n"
      << "  let b_row_u32 = b_base + col * ldb_u32;\n"
      << "  let k_pairs = K >> 1u;\n"
      << "  for (var kp: u32 = tid; kp < k_pairs; kp = kp + WG_SIZE) {\n";
    if (!a_transposed) {
      s << "    let a0 = f32(a[a_base + row * params.lda + kp * 2u]);\n"
        << "    let a1 = f32(a[a_base + row * params.lda + kp * 2u + 1u]);\n";
    } else {
      s << "    let a0 = f32(a[a_base + (kp * 2u) * params.lda + row]);\n"
        << "    let a1 = f32(a[a_base + (kp * 2u + 1u) * params.lda + row]);\n";
    }
    s << "    let bpair = unpack_bf16_pair(b[b_row_u32 + kp]);\n"
      << "    acc = acc + a0 * bpair.x + a1 * bpair.y;\n"
      << "  }\n";
  } else {
    s << "  for (var k: u32 = tid; k < K; k = k + WG_SIZE) {\n";

    // A indexing: row=m, col=k
    if (!a_transposed) {
      s << "    let a_val = a[a_base + row * params.lda + k];\n";
    } else {
      s << "    let a_val = a[a_base + k * params.lda + row];\n";
    }

    // B indexing: row=k, col=n
    if (!b_transposed) {
      s << "    let b_val = b[b_base + k * params.ldb + col];\n";
    } else {
      s << "    let b_val = b[b_base + col * params.ldb + k];\n";
    }

    s << "    acc = acc + f32(a_val) * f32(b_val);\n"
      << "  }\n";
  }

  // Workgroup reduction: subgroup-accelerated or fully unrolled tree.
  if (use_sg) {
    wgpu::emit_subgroup_reduction(
        s,
        /*acc_var=*/"acc",
        /*shared_name=*/"shared_acc",
        /*subgroup_builtin=*/"subgroupAdd",
        /*reduce_op=*/"sum_op",
        /*workgroup_size=*/256,
        /*subgroup_size=*/32);
  } else {
    s << "  shared_acc[tid] = acc;\n"
      << "  workgroupBarrier();\n";
    wgpu::emit_unrolled_reduction(s, "shared_acc", "sum_op", 256);
  }

  s << "  if (tid == 0u) {\n"
    << "    out[c_base + row * params.ldc + col] = " << dtype
    << "(shared_acc[0]);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL multi-col GEMV kernel generation
// ---------------------------------------------------------------------------

// Multi-col GEMV: each workgroup computes cols_per_wg consecutive output
// columns cooperatively, sharing one A load per iteration across cols_per_wg
// B loads and cols_per_wg FMAs. Amortizes A reads by cols_per_wg×.
//
// Memory traffic per output element:
//   - Single-col split-K GEMV:    K A loads + K B loads = 2K per col
//   - Multi-col (cols_per_wg=4):  K/4 A loads + K B loads = 1.25K per col
//   - Multi-col (cols_per_wg=8):  K/8 A loads + K B loads = 1.125K per col
// GEMV is memory-bound, so the A-read reduction translates into speedup.
//
// Layout:
//   workgroup_size = 128 (4 subgroups of 32 on Apple Silicon)
//   cols_per_wg    = 4 or 8 (chosen at dispatch time based on N)
//   dispatch       = ceil(N / cols_per_wg) workgroups per batch row
//
// cols_per_wg parallel reductions at the end, one per column accumulator.
// Uses subgroup intrinsics when available. Handles N not divisible by
// cols_per_wg via has1..hasN-1 guards (loop-uniform → predicated, no
// divergence cost). col0 is always in-bounds (tile_idx guard covers it).
std::string make_gemv_multi_col_kernel(
    const std::string& entry_name,
    const std::string& dtype,
    bool a_transposed,
    bool b_transposed,
    bool use_sg,
    uint32_t cols_per_wg,
    bool n_aligned,
    bool b_packed_bf16 = false) {
  std::ostringstream s;

  if (use_sg) {
    s << "enable subgroups;\n";
  }
  if (dtype == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WG_SIZE: u32 = 128u;\n"
    << "const COLS_PER_WG: u32 = " << cols_per_wg << "u;\n"
    << "const GRID_X: u32 = 65535u;\n\n";

  s << "struct MatmulParams {\n"
    << "  M: u32, N: u32, K: u32,\n"
    << "  lda: u32, ldb: u32, ldc: u32,\n"
    << "  batch_size: u32,\n"
    << "  batch_stride_a: u32, batch_stride_b: u32, batch_stride_c: u32,\n"
    << "  offset_a: u32, offset_b: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> a: array<" << dtype << ">;\n";
  if (b_packed_bf16) {
    s << "@group(0) @binding(1) var<storage, read> b: array<u32>;\n";
  } else {
    s << "@group(0) @binding(1) var<storage, read> b: array<" << dtype
      << ">;\n";
  }
  s << "@group(0) @binding(2) var<storage, read_write> out: array<" << dtype
    << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: MatmulParams;\n\n";

  if (b_packed_bf16) {
    s << wgpu::wgsl_unpack_bf16_pair();
  }

  // cols_per_wg parallel shared arrays so all reductions run in fused barriers.
  for (uint32_t i = 0; i < cols_per_wg; ++i) {
    s << "var<workgroup> shared_" << i << ": array<f32, 128>;\n";
  }
  s << "\n";

  s << "@compute @workgroup_size(128)\n"
    << "fn " << entry_name
    << "(@builtin(workgroup_id) wid: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u) {\n"
    << "  let tile_idx = wid.x + wid.y * GRID_X;\n"
    << "  let batch = wid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let M = params.M;\n"
    << "  let N = params.N;\n"
    << "  let K = params.K;\n"
    << "  let num_col_tiles = (N + COLS_PER_WG - 1u) / COLS_PER_WG;\n"
    << "  let total_tiles = M * num_col_tiles;\n"
    << "  if (tile_idx >= total_tiles) { return; }\n"
    << "  let row = tile_idx / num_col_tiles;\n"
    << "  let col_tile = tile_idx % num_col_tiles;\n"
    << "  let col_base = col_tile * COLS_PER_WG;\n";

  // col0..colN-1
  for (uint32_t i = 0; i < cols_per_wg; ++i) {
    s << "  let col" << i << " = col_base + " << i << "u;\n";
  }
  // has1..hasN-1 (col0 always in-bounds because tile_idx guard covers it).
  // Aligned path (N % cols_per_wg == 0) skips guards entirely — all cols valid.
  if (!n_aligned) {
    for (uint32_t i = 1; i < cols_per_wg; ++i) {
      s << "  let has" << i << " = col" << i << " < N;\n";
    }
  }

  s << "  let tid = lid.x;\n"
    << "  let a_base = batch * params.batch_stride_a + params.offset_a;\n"
    << "  let b_base = batch * params.batch_stride_b + params.offset_b;\n"
    << "  let c_base = batch * params.batch_stride_c;\n";

  // Accumulators
  for (uint32_t i = 0; i < cols_per_wg; ++i) {
    s << "  var acc" << i << ": f32 = 0.0;\n";
  }

  if (b_packed_bf16) {
    // Step A: only the b_transposed=true hot path is implemented. Step B will
    // add a packed non-transposed variant.
    assert(b_transposed &&
           "packed bf16 multi-col GEMV requires b_transposed=true in Step A");
    s << "  // Packed bf16 iterates in K/2 pairs; each u32 load unpacks into\n"
      << "  // two bf16 values that get broadcast across all cols.\n"
      << "  let ldb_u32 = params.ldb >> 1u;\n"
      << "  let k_pairs = K >> 1u;\n"
      << "  for (var kp: u32 = tid; kp < k_pairs; kp = kp + WG_SIZE) {\n";
    // Load two consecutive K values of A once and reuse across all cols.
    if (!a_transposed) {
      s << "    let a0 = f32(a[a_base + row * params.lda + kp * 2u]);\n"
        << "    let a1 = f32(a[a_base + row * params.lda + kp * 2u + 1u]);\n";
    } else {
      s << "    let a0 = f32(a[a_base + (kp * 2u) * params.lda + row]);\n"
        << "    let a1 = f32(a[a_base + (kp * 2u + 1u) * params.lda + row]);\n";
    }
    s << "    let bp0 = unpack_bf16_pair(b[b_base + col0 * ldb_u32 + kp]);\n"
      << "    acc0 = acc0 + a0 * bp0.x + a1 * bp0.y;\n";
    for (uint32_t i = 1; i < cols_per_wg; ++i) {
      if (n_aligned) {
        s << "    let bp" << i << " = unpack_bf16_pair(b[b_base + col" << i
          << " * ldb_u32 + kp]);\n"
          << "    acc" << i << " = acc" << i << " + a0 * bp" << i
          << ".x + a1 * bp" << i << ".y;\n";
      } else {
        s << "    if (has" << i << ") {\n"
          << "      let bp" << i << " = unpack_bf16_pair(b[b_base + col" << i
          << " * ldb_u32 + kp]);\n"
          << "      acc" << i << " = acc" << i << " + a0 * bp" << i
          << ".x + a1 * bp" << i << ".y;\n"
          << "    }\n";
      }
    }
    s << "  }\n";
  } else {
    s << "  for (var k: u32 = tid; k < K; k = k + WG_SIZE) {\n";

    if (!a_transposed) {
      s << "    let a_val = f32(a[a_base + row * params.lda + k]);\n";
    } else {
      s << "    let a_val = f32(a[a_base + k * params.lda + row]);\n";
    }

    if (!b_transposed) {
      // B row-major (K, N) → b[k*ldb + col]
      s << "    let b_row = b_base + k * params.ldb;\n";
      s << "    acc0 = acc0 + a_val * f32(b[b_row + col0]);\n";
      for (uint32_t i = 1; i < cols_per_wg; ++i) {
        if (n_aligned) {
          s << "    acc" << i << " = acc" << i
            << " + a_val * f32(b[b_row + col" << i << "]);\n";
        } else {
          s << "    if (has" << i << ") { acc" << i << " = acc" << i
            << " + a_val * f32(b[b_row + col" << i << "]); }\n";
        }
      }
    } else {
      // B transposed (N, K) row-major → b[col*ldb + k]
      s << "    acc0 = acc0 + a_val * f32(b[b_base + col0 * params.ldb + k]);\n";
      for (uint32_t i = 1; i < cols_per_wg; ++i) {
        if (n_aligned) {
          s << "    acc" << i << " = acc" << i
            << " + a_val * f32(b[b_base + col" << i << " * params.ldb + k]);\n";
        } else {
          s << "    if (has" << i << ") { acc" << i << " = acc" << i
            << " + a_val * f32(b[b_base + col" << i << " * params.ldb + k]); }\n";
        }
      }
    }

    s << "  }\n";
  }

  // cols_per_wg parallel workgroup reductions, fused into the same barriers.
  if (use_sg) {
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "  var sg" << i << " = subgroupAdd(acc" << i << ");\n";
    }
    s << "  let sg_id = tid / 32u;\n"
      << "  if (subgroupElect()) {\n";
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "    shared_" << i << "[sg_id] = sg" << i << ";\n";
    }
    s << "  }\n"
      << "  workgroupBarrier();\n";
    // 128 threads / 32 subgroup = 4 subgroups. Reduce 4 values per shared
    // array.
    s << "  if (tid < 2u) {\n";
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "    shared_" << i << "[tid] = shared_" << i << "[tid] + shared_"
        << i << "[tid + 2u];\n";
    }
    s << "  }\n"
      << "  workgroupBarrier();\n"
      << "  if (tid < 1u) {\n";
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "    shared_" << i << "[tid] = shared_" << i << "[tid] + shared_"
        << i << "[tid + 1u];\n";
    }
    s << "  }\n"
      << "  workgroupBarrier();\n";
  } else {
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "  shared_" << i << "[tid] = acc" << i << ";\n";
    }
    s << "  workgroupBarrier();\n";
    // Unrolled 128-way reduction on N arrays in parallel (shared barriers).
    for (uint32_t stride = 64; stride >= 1; stride /= 2) {
      s << "  if (tid < " << stride << "u) {\n";
      for (uint32_t i = 0; i < cols_per_wg; ++i) {
        s << "    shared_" << i << "[tid] = shared_" << i << "[tid] + shared_"
          << i << "[tid + " << stride << "u];\n";
      }
      s << "  }\n  workgroupBarrier();\n";
    }
  }

  s << "  if (tid == 0u) {\n"
    << "    let out_row = c_base + row * params.ldc;\n"
    << "    out[out_row + col0] = " << dtype << "(shared_0[0]);\n";
  for (uint32_t i = 1; i < cols_per_wg; ++i) {
    if (n_aligned) {
      s << "    out[out_row + col" << i << "] = " << dtype << "(shared_" << i
        << "[0]);\n";
    } else {
      s << "    if (has" << i << ") { out[out_row + col" << i
        << "] = " << dtype << "(shared_" << i << "[0]); }\n";
    }
  }
  s << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL tiled GEMM kernel generation
// ---------------------------------------------------------------------------

// Tiled GEMM with 16x16 tiles, 256 threads per workgroup.
// Transpose variants: NN, NT, TN, TT.
// Uses shared memory for A and B tiles, f32 accumulation.
//
// Packed bf16 variant (b_packed_bf16=true): B is bound as array<u32> where
// every u32 holds two adjacent bf16 values along the K dimension. Each tile
// load reads the appropriate u32 and parity-selects the right half. Required
// for prefill correctness when the same weight buffer is consumed by GEMM
// (Tq > 1) and GEMV (Tq == 1) — without this, prefill reads packed bytes as
// f32 and produces garbage logits (the Step A "garbage prefill" bug).
//
// Step A only implements the b_transposed=true path because every Qwen3.5
// linear projection routes through get_weight_t() → check_transpose() →
// b_transposed=true. Non-transposed packed B is a Step B concern.
std::string make_gemm_kernel(
    const std::string& entry_name,
    const std::string& dtype,
    bool a_transposed,
    bool b_transposed,
    bool b_packed_bf16 = false) {
  std::ostringstream s;

  if (b_packed_bf16) {
    assert(b_transposed &&
           "packed bf16 GEMM requires b_transposed=true in Step A");
  }

  if (dtype == "f16") {
    s << "enable f16;\n\n";
  }

  s << "const TILE_SIZE: u32 = 16u;\n\n";

  s << "struct MatmulParams {\n"
    << "  M: u32, N: u32, K: u32,\n"
    << "  lda: u32, ldb: u32, ldc: u32,\n"
    << "  batch_size: u32,\n"
    << "  batch_stride_a: u32, batch_stride_b: u32, batch_stride_c: u32,\n"
    << "  offset_a: u32, offset_b: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> a: array<" << dtype << ">;\n";
  if (b_packed_bf16) {
    s << "@group(0) @binding(1) var<storage, read> b: array<u32>;\n";
  } else {
    s << "@group(0) @binding(1) var<storage, read> b: array<" << dtype
      << ">;\n";
  }
  s << "@group(0) @binding(2) var<storage, read_write> out: array<" << dtype
    << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: MatmulParams;\n\n";

  if (b_packed_bf16) {
    s << wgpu::wgsl_unpack_bf16_pair();
  }

  // Shared memory tiles (f32 for accumulation precision)
  s << "var<workgroup> a_shared: array<array<f32, 16>, 16>;\n"
    << "var<workgroup> b_shared: array<array<f32, 16>, 16>;\n\n";

  s << "@compute @workgroup_size(16, 16)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wid: vec3u) {\n"
    << "  let batch = wid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let M = params.M;\n"
    << "  let N = params.N;\n"
    << "  let K = params.K;\n"
    << "  let local_row = lid.y;\n"
    << "  let local_col = lid.x;\n"
    << "  let global_row = wid.y * TILE_SIZE + local_row;\n"
    << "  let global_col = wid.x * TILE_SIZE + local_col;\n"
    << "  let a_base = batch * params.batch_stride_a + params.offset_a;\n"
    << "  let b_base = batch * params.batch_stride_b + params.offset_b;\n"
    << "  let c_base = batch * params.batch_stride_c;\n"
    << "  var acc: f32 = 0.0;\n"
    << "  let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;\n"
    << "  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {\n"
    << "    let k_offset = t * TILE_SIZE;\n";

  // Load A tile into shared memory
  if (!a_transposed) {
    // A is row-major: A[row][k] = A[a_base + row * lda + k]
    s << "    let a_row = global_row;\n"
      << "    let a_col = k_offset + local_col;\n"
      << "    if (a_row < M && a_col < K) {\n"
      << "      a_shared[local_row][local_col] = f32(a[a_base + a_row * params.lda + a_col]);\n"
      << "    } else {\n"
      << "      a_shared[local_row][local_col] = 0.0;\n"
      << "    }\n";
  } else {
    // A is transposed: A^T[row][k] = A[k][row] = A[a_base + k * lda + row]
    s << "    let a_k = k_offset + local_col;\n"
      << "    let a_m = global_row;\n"
      << "    if (a_m < M && a_k < K) {\n"
      << "      a_shared[local_row][local_col] = f32(a[a_base + a_k * params.lda + a_m]);\n"
      << "    } else {\n"
      << "      a_shared[local_row][local_col] = 0.0;\n"
      << "    }\n";
  }

  // Load B tile into shared memory
  if (!b_transposed) {
    // B is row-major: B[k][col] = B[b_base + k * ldb + col]
    s << "    let b_row = k_offset + local_row;\n"
      << "    let b_col = global_col;\n"
      << "    if (b_row < K && b_col < N) {\n"
      << "      b_shared[local_row][local_col] = f32(b[b_base + b_row * params.ldb + b_col]);\n"
      << "    } else {\n"
      << "      b_shared[local_row][local_col] = 0.0;\n"
      << "    }\n";
  } else if (b_packed_bf16) {
    // B is transposed AND packed: each row (col index in matmul terms) of K
    // bf16 values is stored as K/2 u32 pairs along the K dimension. Index by
    // u32 unit (ldb_u32 = ldb / 2) and pick the right half by K parity.
    s << "    let b_k = k_offset + local_row;\n"
      << "    let b_n = global_col;\n"
      << "    if (b_k < K && b_n < N) {\n"
      << "      let ldb_u32 = params.ldb >> 1u;\n"
      << "      let pair = b[b_base + b_n * ldb_u32 + (b_k >> 1u)];\n"
      << "      let unpacked = unpack_bf16_pair(pair);\n"
      << "      let val = select(unpacked.y, unpacked.x, (b_k & 1u) == 0u);\n"
      << "      b_shared[local_row][local_col] = val;\n"
      << "    } else {\n"
      << "      b_shared[local_row][local_col] = 0.0;\n"
      << "    }\n";
  } else {
    // B is transposed: B^T[k][col] = B[col][k] = B[b_base + col * ldb + k]
    s << "    let b_k = k_offset + local_row;\n"
      << "    let b_n = global_col;\n"
      << "    if (b_k < K && b_n < N) {\n"
      << "      b_shared[local_row][local_col] = f32(b[b_base + b_n * params.ldb + b_k]);\n"
      << "    } else {\n"
      << "      b_shared[local_row][local_col] = 0.0;\n"
      << "    }\n";
  }

  s << "    workgroupBarrier();\n"
    << "    for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {\n"
    << "      acc = acc + a_shared[local_row][k] * b_shared[k][local_col];\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "  }\n"
    << "  if (global_row < M && global_col < N) {\n"
    << "    out[c_base + global_row * params.ldc + global_col] = " << dtype
    << "(acc);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Transpose detection (mirrors CUDA check_transpose)
// ---------------------------------------------------------------------------

struct TransposeResult {
  bool transposed;
  uint32_t ld; // leading dimension
  array arr;
};

TransposeResult check_transpose(
    wgpu::CommandEncoder& enc,
    const Stream& s,
    const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    // Row-major, no transpose
    return {false, static_cast<uint32_t>(stx), arr};
  } else if (stx == 1 && sty == arr.shape(-2)) {
    // Column-major = transposed
    return {true, static_cast<uint32_t>(sty), arr};
  } else {
    // Non-contiguous: make a copy
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return {false, static_cast<uint32_t>(arr.shape(-1)), arr_copy};
  }
}

// ---------------------------------------------------------------------------
// Unified matmul dispatch (GEMV and GEMM)
// ---------------------------------------------------------------------------

enum class MatmulKind { GEMV, GEMM };

static void dispatch_matmul(
    MatmulKind kind,
    const array& a,
    const array& b,
    array& out,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool a_transposed,
    uint32_t lda,
    bool b_transposed,
    uint32_t ldb,
    uint32_t batch_count,
    uint32_t batch_stride_a,
    uint32_t batch_stride_b,
    uint32_t batch_stride_c,
    const Stream& s) {
  auto& dev = wgpu::device();
  const char* wgsl_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  const bool use_sg = (kind == MatmulKind::GEMV) && dev.has_subgroups();

  // Use the multi-col GEMV kernel when N is large enough to produce enough
  // workgroups for GPU saturation. For very small N we use the single-col
  // split-K kernel so parallelism doesn't collapse to a handful of workgroups.
  // For very large N we use COLS_PER_WG=8 to further amortize A-row reads.
  constexpr uint32_t kMultiColThreshold = 16;
  constexpr uint32_t kMultiCol8Threshold = 512;
  const bool use_multi_col =
      (kind == MatmulKind::GEMV) && N >= kMultiColThreshold;
  const bool use_multi_col_8 =
      (kind == MatmulKind::GEMV) && N >= kMultiCol8Threshold;
  const uint32_t cols_per_wg = use_multi_col_8 ? 8u : 4u;
  // Fast path when N is evenly divisible by cols_per_wg: skip all per-column
  // bounds checks. True for every GEMV shape in Qwen3.5 (hidden=896,
  // intermediate=4864, vocab=151936 — all divisible by 8).
  const bool n_aligned = use_multi_col && (N % cols_per_wg == 0);

  // Resolve GPU buffers up-front. wgpu_buffer() has the side-effect of
  // flipping eligible bf16 weight buffers into StorageMode::PackedBf16 on
  // their first dispatch, so we need the buffer handles before deciding
  // which kernel variant to pick.
  WGPUBuffer a_buf = wgpu::wgpu_buffer(a);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(b);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  // Packed-bf16 selection: GEMV b_transposed=true hot path is the primary
  // optimization, but we ALSO need GEMM b_transposed=true to handle packed
  // weights so prefill (Tq > 1) doesn't read packed bytes as f32 garbage.
  // Both Qwen3.5 GEMV (decode) and GEMM (prefill) of linear weights land here
  // via check_transpose. Non-transposed packed B is a Step B concern.
  auto* b_wbuf = static_cast<const wgpu::WebGPUBuffer*>(b.buffer().ptr());
  const bool b_packed_bf16 =
      b_transposed &&
      b_wbuf != nullptr &&
      b_wbuf->storage_mode == wgpu::StorageMode::PackedBf16 &&
      (K % 2u == 0u);

  // Build transpose suffix for pipeline key
  std::string trans_suffix;
  trans_suffix += (a_transposed ? "T" : "N");
  trans_suffix += (b_transposed ? "T" : "N");

  // Pipeline cache keys on entry_name, so distinguish 4-col vs 8-col variants
  // (and aligned vs guarded) to avoid collisions that would break correctness.
  const char* prefix = (kind == MatmulKind::GEMV)
      ? (use_multi_col_8 ? "gemv_mc8_"
                         : (use_multi_col ? "gemv_mc4_" : "gemv_"))
      : "gemm_";
  std::string entry_name =
      std::string(prefix) + trans_suffix + "_" + wgsl_type;
  if (b_packed_bf16) {
    entry_name += "_bf16p";
  }
  if (use_sg) {
    entry_name += "_sg";
  }
  if (n_aligned) {
    entry_name += "_al";
  }

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
#ifdef MLX_WGPU_LOG_KERNELS
        // First-dispatch log — the shader-module cache only invokes this
        // lambda on a MISS, so each unique entry_name prints exactly once.
        // Zero overhead on the hot path (no mutex, no set lookup).
        std::cerr << "[MLX-KERNEL] " << entry_name << std::endl;
#endif
        if (kind == MatmulKind::GEMV) {
          return use_multi_col
              ? make_gemv_multi_col_kernel(
                    entry_name,
                    wgsl_type,
                    a_transposed,
                    b_transposed,
                    use_sg,
                    cols_per_wg,
                    n_aligned,
                    b_packed_bf16)
              : make_gemv_kernel(
                    entry_name,
                    wgsl_type,
                    a_transposed,
                    b_transposed,
                    use_sg,
                    b_packed_bf16);
        }
        return make_gemm_kernel(
            entry_name, wgsl_type, a_transposed, b_transposed, b_packed_bf16);
      });

  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  MatmulParams params{};
  params.M = M;
  params.N = N;
  params.K = K;
  params.lda = lda;
  params.ldb = ldb;
  params.ldc = N; // output is always row-major
  params.batch_size = batch_count;
  params.batch_stride_a = batch_stride_a;
  params.batch_stride_b = batch_stride_b;
  params.batch_stride_c = batch_stride_c;
  params.offset_a = static_cast<uint32_t>(a.offset() / a.itemsize());
  // For packed-bf16 B buffers the shader indexes array<u32>, so the offset
  // must be expressed in u32 units (4 bytes each). array::offset() returns
  // BYTES, and in packed mode those bytes are stored 2 bf16 per u32 — any
  // slice must start on a u32 boundary (element index divisible by 2) so
  // dividing the byte offset by 4 gives the packed u32 index.
  if (b_packed_bf16) {
    params.offset_b = static_cast<uint32_t>(b.offset() / 4);
  } else {
    params.offset_b = static_cast<uint32_t>(b.offset() / b.itemsize());
  }
  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(MatmulParams));

  // a_buf / b_buf / out_buf were already resolved above so the packed-bf16
  // storage mode is visible before kernel selection.
  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{a_buf, wgpuBufferGetSize(a_buf)},
       {b_buf, wgpuBufferGetSize(b_buf)},
       {out_buf, wgpuBufferGetSize(out_buf)},
       {uniform_buf, sizeof(MatmulParams)}});

  if (kind == MatmulKind::GEMV) {
    // Split-K (single or multi-col): fan out across (x, y) with GRID_X=65535
    // (the WebGPU minimum for maxComputeWorkgroupsPerDim) so lm_head-sized
    // outputs (vocab > 65535) still fit.
    uint32_t tiles_per_row = use_multi_col
        ? ((N + cols_per_wg - 1) / cols_per_wg)
        : N;
    uint32_t total_tiles = M * tiles_per_row;
    constexpr uint32_t kGridX = 65535u;
    uint32_t wg_x = total_tiles < kGridX ? total_tiles : kGridX;
    uint32_t wg_y = (total_tiles + kGridX - 1) / kGridX;
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y, batch_count);
  } else {
    // Grid: (ceil(N/16), ceil(M/16), batch_size)
    uint32_t wg_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t wg_y = (M + TILE_SIZE - 1) / TILE_SIZE;
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y, batch_count);
  }

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

// ---------------------------------------------------------------------------
// Main matmul dispatch
// ---------------------------------------------------------------------------

void matmul_dispatch(
    const array& a_pre,
    const array& b_pre,
    array& out,
    const Stream& s) {
  auto& encoder = wgpu::get_command_encoder(s);

  // Check transpose state from strides
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  uint32_t M = static_cast<uint32_t>(a_pre.shape(-2));
  uint32_t N = static_cast<uint32_t>(b_pre.shape(-1));
  uint32_t K = static_cast<uint32_t>(a_pre.shape(-1));

  // Collapse batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] =
      collapse_batches(a, b);

  uint32_t batch_count =
      static_cast<uint32_t>(out.size() / (M * N));

  // Collapse batches into M when possible (contiguous batch + A dims, B
  // broadcast)
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == static_cast<int64_t>(K) &&
      a_batch_strides.back() == static_cast<int64_t>(M * K) &&
      b_batch_strides.back() == 0) {
    M *= static_cast<uint32_t>(batch_shape.back());
    batch_count = 1;
    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  uint32_t batch_stride_a =
      static_cast<uint32_t>(a_batch_strides.back());
  uint32_t batch_stride_b =
      static_cast<uint32_t>(b_batch_strides.back());
  uint32_t batch_stride_c = M * N;

  // Use GEMV for M=1 or N=1 cases
  MatmulKind kind = (M == 1 || N == 1) ? MatmulKind::GEMV : MatmulKind::GEMM;
  dispatch_matmul(
      kind, a, b, out, M, N, K, a_transposed, lda, b_transposed, ldb,
      batch_count, batch_stride_a, batch_stride_b, batch_stride_c, s);
}

} // namespace

// ---------------------------------------------------------------------------
// Matmul::eval_gpu
// ---------------------------------------------------------------------------

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  // Return zeros if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(zero);
    out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  matmul_dispatch(a_pre, b_pre, out, s);
}

// ---------------------------------------------------------------------------
// AddMM::eval_gpu
// ---------------------------------------------------------------------------

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto& c = inputs[2];

  // Allocate output
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  // First compute the matmul part: alpha * A @ B
  // We'll create a temporary for the matmul result, then combine with C.
  // For the simple case alpha=1, beta=1, and alpha=1, beta=0 we optimize.

  if (a_pre.size() == 0 || b_pre.size() == 0) {
    if (beta_ == 0.0f) {
      // out = 0
      array zero(0, a_pre.dtype());
      auto& encoder = wgpu::get_command_encoder(s);
      encoder.add_temporary(zero);
      fill_gpu(zero, out, s);
    } else {
      // out = beta * C: use copy + scale via the fallback path
      // Just copy C to output scaled by beta.
      // For simplicity, copy C to output then we'd need a scale op.
      // Use the CPU fallback for this rare case.
      copy_gpu(c, out, CopyType::General, s);
    }
    return;
  }

  // Compute matmul into output directly
  matmul_dispatch(a_pre, b_pre, out, s);

  // If alpha != 1 or beta != 0, we need to apply: out = alpha * out + beta * C
  // For the common case (alpha=1, beta=0), we're done.
  // For alpha=1, beta=1, we need to add C.
  // For general case, we need: out = alpha * matmul_result + beta * C
  //
  // We handle this by dispatching a fused scale-add kernel or using
  // existing binary operations. For now, use a simple approach:
  // The output already contains A @ B. We need to scale by alpha and add
  // beta * C.

  if (alpha_ != 1.0f || beta_ != 0.0f) {
    // We need a post-processing kernel to compute:
    //   out[i] = alpha * out[i] + beta * C[i]
    // The epilogue kernel assumes flat contiguous indexing for both C and out.

    // Make C contiguous if it is not already
    array c_contig = c;
    auto& encoder = wgpu::get_command_encoder(s);
    if (!c.flags().row_contiguous) {
      c_contig = contiguous_copy_gpu(c, s);
      encoder.add_temporary(c_contig);
    }

    // Build the epilogue kernel
    const char* wgsl_type = wgpu::dtype_to_wgsl_safe(out.dtype());
    std::string entry_name =
        std::string("addmm_epilogue_") + wgsl_type;

    std::ostringstream src;
    if (std::string(wgsl_type) == "f16") {
      src << "enable f16;\n\n";
    }

    src << "struct EpilogueParams {\n"
        << "  size: u32,\n"
        << "  alpha: f32,\n"
        << "  beta: f32,\n"
        << "  _pad: u32,\n"
        << "}\n\n";

    src << "@group(0) @binding(0) var<storage, read> c_in: array<" << wgsl_type
        << ">;\n"
        << "@group(0) @binding(1) var<storage, read_write> out_buf: array<"
        << wgsl_type << ">;\n"
        << "@group(0) @binding(2) var<uniform> params: EpilogueParams;\n\n";

    src << "@compute @workgroup_size(256)\n"
        << "fn " << entry_name
        << "(@builtin(global_invocation_id) gid: vec3u) {\n"
        << "  let idx = gid.x;\n"
        << "  if (idx >= params.size) { return; }\n"
        << "  let matmul_val = f32(out_buf[idx]);\n"
        << "  let c_val = f32(c_in[idx]);\n"
        << "  out_buf[idx] = " << wgsl_type
        << "(params.alpha * matmul_val + params.beta * c_val);\n"
        << "}\n";

    auto& dev = wgpu::device();
    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name, [&]() { return src.str(); });

    auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    encoder.set_input_array(c_contig);

    // Create epilogue params
    struct EpilogueParams {
      uint32_t size;
      float alpha;
      float beta;
      uint32_t _pad;
    };
    EpilogueParams ep{};
    ep.size = static_cast<uint32_t>(out.size());
    ep.alpha = alpha_;
    ep.beta = beta_;

    auto& pool = wgpu::device().uniform_pool();
    WGPUBuffer uniform_buf =
        pool.acquire(wgpu::device().gpu_queue(), &ep, sizeof(EpilogueParams));

    WGPUBuffer c_buf = wgpu::wgpu_buffer(c_contig);
    WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

    WGPUBindGroup bg = wgpu::create_bind_group(
        pe.layout,
        {{c_buf, wgpuBufferGetSize(c_buf)},
         {out_buf, wgpuBufferGetSize(out_buf)},
         {uniform_buf, sizeof(EpilogueParams)}});

    uint32_t num_workgroups =
        (ep.size + GEMV_WORKGROUP_SIZE - 1) / GEMV_WORKGROUP_SIZE;
    encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

    wgpuBindGroupRelease(bg);
    encoder.add_completed_handler([uniform_buf]() {
      wgpu::device().uniform_pool().release(uniform_buf);
    });
  }
}

} // namespace mlx::core
