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
#include <cstdint>
#include <cstring>
#include <limits>
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

// Layout must match the WGSL struct emitted by wgsl_matmul_params_struct().
// vec4<u32> fields require 16-byte alignment in WGSL uniform buffers; the
// first vec4 (batch_shape) starts at offset 32 which satisfies this.
//
// MM-F012 safe-model-size note: all offsets / strides in this struct are u32.
// `offset_a`, `offset_b`, and `batch_stride_*` are therefore limited to
// 2^32 - 1 *elements* (or u32 indices, for the packed bf16 B path). For f32
// weights that is ~16 GiB worth of elements; for bf16 weights stored packed
// (two per u32) that is ~16 Gi u32 pairs ≈ 32 Gi bf16 scalars. In practice
// the biggest buffer we touch today (Qwen3.5-0.8B embeddings) is ~272 MiB and
// sits comfortably inside u32 range. dispatch_matmul() has explicit runtime
// asserts that every offset/stride it pushes fits in u32 so an oversized
// model fails loudly instead of silently truncating into a garbage index.
struct MatmulParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
  uint32_t batch_size;
  uint32_t batch_ndim;
  uint32_t batch_shape[4];      // maps to vec4<u32> in WGSL
  uint32_t batch_stride_a[4];   // maps to vec4<u32> in WGSL
  uint32_t batch_stride_b[4];   // maps to vec4<u32> in WGSL
  uint32_t batch_stride_c;      // output is always contiguous
  uint32_t offset_a;
  uint32_t offset_b;
  uint32_t _pad;                // pad to multiple of 16 bytes
};
static_assert(sizeof(MatmulParams) == 96, "MatmulParams must be 96 bytes to match WGSL layout");

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
    int subgroup_size,
    bool b_packed_bf16 = false) {
  const bool use_sg = subgroup_size > 0;
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

  s << wgpu::wgsl_matmul_params_struct();

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
    << "  // MM-F006 contract: `output_idx = wid.x + wid.y * GRID_X` assumes\n"
    << "  // the host decomposed `total_tiles` using `wg_x == GRID_X` (65535)\n"
    << "  // whenever `wg_y > 1`. Any other `wg_x` with `wg_y > 1` produces a\n"
    << "  // non-bijective mapping and silent wrong logits. The host-side\n"
    << "  // invariant is asserted in dispatch_matmul() below.\n"
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
    << "  let batch_locs = elem_to_loc_broadcast(batch, params.batch_ndim, params.batch_shape, params.batch_stride_a, params.batch_stride_b);\n"
    << "  let a_base = batch_locs.x + params.offset_a;\n"
    << "  let b_base = batch_locs.y + params.offset_b;\n"
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
        /*subgroup_size=*/static_cast<uint32_t>(subgroup_size));
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
    int subgroup_size,
    uint32_t cols_per_wg,
    bool n_aligned,
    bool b_packed_bf16 = false) {
  const bool use_sg = subgroup_size > 0;
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

  s << wgpu::wgsl_matmul_params_struct();

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
    << "  // MM-F006 contract: `tile_idx = wid.x + wid.y * GRID_X` assumes the\n"
    << "  // host decomposed `total_tiles` using `wg_x == GRID_X` (65535)\n"
    << "  // whenever `wg_y > 1`. Any other `wg_x` with `wg_y > 1` produces a\n"
    << "  // non-bijective mapping. The host-side invariant is asserted in\n"
    << "  // dispatch_matmul() below.\n"
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
    << "  let batch_locs = elem_to_loc_broadcast(batch, params.batch_ndim, params.batch_shape, params.batch_stride_a, params.batch_stride_b);\n"
    << "  let a_base = batch_locs.x + params.offset_a;\n"
    << "  let b_base = batch_locs.y + params.offset_b;\n"
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
    const uint32_t sg = static_cast<uint32_t>(subgroup_size);
    const uint32_t num_partials = 128u / sg; // 128-thread workgroup
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "  var sg" << i << " = subgroupAdd(acc" << i << ");\n";
    }
    s << "  let sg_id = tid / " << sg << "u;\n"
      << "  if (subgroupElect()) {\n";
    for (uint32_t i = 0; i < cols_per_wg; ++i) {
      s << "    shared_" << i << "[sg_id] = sg" << i << ";\n";
    }
    s << "  }\n"
      << "  workgroupBarrier();\n";
    // Tree-reduce the `num_partials` subgroup results across shared memory.
    // num_partials = 128 / subgroup_size: 4 (sg=32), 2 (sg=64), 8 (sg=16),
    // 16 (sg=8), 1 (sg=128). Emit a codegen-time loop so the WGSL is
    // correct for every detected subgroup_size, not just sg=32.
    for (uint32_t stride = num_partials >> 1; stride >= 1; stride >>= 1) {
      s << "  if (tid < " << stride << "u) {\n";
      for (uint32_t i = 0; i < cols_per_wg; ++i) {
        s << "    shared_" << i << "[tid] = shared_" << i
          << "[tid] + shared_" << i << "[tid + " << stride << "u];\n";
      }
      s << "  }\n"
        << "  workgroupBarrier();\n";
    }
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

  s << wgpu::wgsl_matmul_params_struct();

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
    << "  let batch_locs = elem_to_loc_broadcast(batch, params.batch_ndim, params.batch_shape, params.batch_stride_a, params.batch_stride_b);\n"
    << "  let a_base = batch_locs.x + params.offset_a;\n"
    << "  let b_base = batch_locs.y + params.offset_b;\n"
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
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    uint32_t batch_stride_c,
    const Stream& s) {
  auto& dev = wgpu::device();
  const char* wgsl_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  // Only GEMV uses subgroup intrinsics; GEMM has its own 16x16-tile reduction.
  const int sg =
      (kind == MatmulKind::GEMV) ? wgpu::effective_subgroup_size() : 0;

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
  // subgroup_suffix() returns "_sg<N>" when the subgroup path is enabled,
  // empty otherwise. Partitions the pipeline cache by detected lane count
  // so a pipeline compiled for 32-lane hardware is never reused on a
  // 64-lane device.
  entry_name += wgpu::subgroup_suffix();
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
                    sg,
                    cols_per_wg,
                    n_aligned,
                    b_packed_bf16)
              : make_gemv_kernel(
                    entry_name,
                    wgsl_type,
                    a_transposed,
                    b_transposed,
                    sg,
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
  if (batch_shape.size() > 4) {
    throw std::runtime_error(
        "[WebGPU matmul] batch_ndim > 4 exceeds vec4 uniform field capacity");
  }
  params.batch_ndim = static_cast<uint32_t>(batch_shape.size());
  // Zero-init arrays, then fill from the vectors
  std::memset(params.batch_shape, 0, sizeof(params.batch_shape));
  std::memset(params.batch_stride_a, 0, sizeof(params.batch_stride_a));
  std::memset(params.batch_stride_b, 0, sizeof(params.batch_stride_b));
  // MM-F012: every stride we push into the uniform is narrowed to u32. For the
  // model sizes we actually dispatch (see MatmulParams comment) this is safe,
  // but fail loudly instead of silently truncating if that ever stops holding.
  constexpr int64_t kU32Max =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  for (size_t i = 0; i < batch_shape.size() && i < 4; ++i) {
    // batch_shape is int32_t so by definition fits in u32 once >= 0. Keep
    // only the sign check — the upper-bound comparison would be tautological
    // and triggers a -Wtautological-constant-out-of-range-compare warning.
    if (batch_shape[i] < 0) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F012: batch_shape entry must be non-negative");
    }
    if (a_batch_strides[i] < 0 || a_batch_strides[i] > kU32Max) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F012: a batch stride does not fit in u32");
    }
    // For packed-bf16 B, the kernel indexes array<u32> (2 bf16 per u32),
    // so batch strides must be in u32 units. collapse_batches returns
    // element strides, so divide by 2. (In practice packed B is always
    // a 2D weight with stride 0, but guard against future batched use.)
    const int64_t b_stride = b_packed_bf16
        ? (b_batch_strides[i] / 2)
        : b_batch_strides[i];
    if (b_stride < 0 || b_stride > kU32Max) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F012: b batch stride does not fit in u32");
    }
    params.batch_shape[i] = static_cast<uint32_t>(batch_shape[i]);
    params.batch_stride_a[i] = static_cast<uint32_t>(a_batch_strides[i]);
    params.batch_stride_b[i] = static_cast<uint32_t>(b_stride);
  }
  params.batch_stride_c = batch_stride_c;
  // MM-F012: offset_a is in element units — verify it survives u32 narrowing.
  const int64_t offset_a_elems =
      static_cast<int64_t>(a.offset()) /
      static_cast<int64_t>(a.itemsize());
  if (offset_a_elems < 0 || offset_a_elems > kU32Max) {
    throw std::runtime_error(
        "[WebGPU matmul] MM-F012: offset_a does not fit in u32");
  }
  params.offset_a = static_cast<uint32_t>(offset_a_elems);
  // For packed-bf16 B buffers the shader indexes array<u32>, so the offset
  // must be expressed in u32 units (4 bytes each). array::offset() returns
  // BYTES, and in packed mode those bytes are stored 2 bf16 per u32 — any
  // slice must start on a u32 boundary (element index divisible by 2) so
  // dividing the byte offset by 4 gives the packed u32 index.
  if (b_packed_bf16) {
    // MM-F011: b.offset() is a byte count. Packed-bf16 indexing divides it by
    // 4 to obtain a u32 index, which silently rounds down any misaligned
    // slice into a wrong starting element. Reject misalignment loudly.
    if ((b.offset() % 4u) != 0u) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F011: packed bf16 b.offset() must be a multiple"
          " of 4 bytes (u32-aligned); got byte offset not divisible by 4");
    }
    const int64_t offset_b_u32 = static_cast<int64_t>(b.offset()) / 4;
    if (offset_b_u32 < 0 || offset_b_u32 > kU32Max) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F012: packed offset_b does not fit in u32");
    }
    params.offset_b = static_cast<uint32_t>(offset_b_u32);
  } else {
    const int64_t offset_b_elems =
        static_cast<int64_t>(b.offset()) /
        static_cast<int64_t>(b.itemsize());
    if (offset_b_elems < 0 || offset_b_elems > kU32Max) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F012: offset_b does not fit in u32");
    }
    params.offset_b = static_cast<uint32_t>(offset_b_elems);
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
    // MM-F006: compute total_tiles in u64 first so we can detect the case
    // where the 2D decomposition itself is undefined. The (wg_x, wg_y) fan
    // out uses a fixed GRID_X=65535, so the addressable space is 65535 *
    // 65535 = 4_294_836_225 tiles. Beyond that, `wg_y` would need to exceed
    // 65535 too, which the WebGPU spec does not guarantee any device will
    // accept (maxComputeWorkgroupsPerDimension minimum is 65535).
    const uint64_t total_tiles_u64 =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(tiles_per_row);
    constexpr uint64_t kGridMax2D =
        static_cast<uint64_t>(65535u) * static_cast<uint64_t>(65535u);
    if (total_tiles_u64 >= kGridMax2D) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F006: total_tiles >= 65535^2; the 2D (wg_x, "
          "wg_y) decomposition cannot address this many tiles with "
          "GRID_X=65535 fan-out");
    }
    const uint32_t total_tiles = static_cast<uint32_t>(total_tiles_u64);
    constexpr uint32_t kGridX = 65535u;
    uint32_t wg_x = total_tiles < kGridX ? total_tiles : kGridX;
    uint32_t wg_y = (total_tiles + kGridX - 1) / kGridX;
    // MM-F006: the GEMV kernels compute `tile_idx = wid.x + wid.y * GRID_X`
    // with GRID_X baked to 65535. The (wid.x, wid.y) -> tile_idx mapping is
    // only bijective when `wg_x == GRID_X` whenever `wg_y > 1`. If we ever
    // relaxed the clamp, short rows would collide with different `wid.y`
    // values and silently produce wrong outputs. Assert the invariant loudly
    // here instead of shipping garbage logits.
    if (wg_y > 1 && wg_x != kGridX) {
      throw std::runtime_error(
          "[WebGPU matmul] MM-F006: GRID_X=65535 dispatch invariant violated"
          " (wg_y > 1 requires wg_x == 65535 so kernel tile_idx stays"
          " bijective)");
    }
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

  uint32_t batch_stride_c = M * N;

  // Use GEMV for M=1 or N=1 cases
  MatmulKind kind = (M == 1 || N == 1) ? MatmulKind::GEMV : MatmulKind::GEMM;
  dispatch_matmul(
      kind, a, b, out, M, N, K, a_transposed, lda, b_transposed, ldb,
      batch_count, batch_shape, a_batch_strides, b_batch_strides,
      batch_stride_c, s);
}

// ---------------------------------------------------------------------------
// MM-F014: pipeline warmup for first-token latency
// ---------------------------------------------------------------------------
//
// Qwen3.5-0.8B decode + prefill together exercise ~6-12 distinct matmul
// variants (GEMV single-col / GEMV mc4 / GEMV mc8 / GEMM NT — each with and
// without packed bf16 B). The first dispatch of each variant invokes the
// WebGPU driver's shader compiler (10-50 ms per pipeline on current
// hardware). When those compiles happen in the hot path they stack directly
// into visible TTFT.
//
// `warmup_matmul_pipelines_impl` populates the shader-module + pipeline cache
// with the variant set actually fired by Qwen3.5-0.8B so the real dispatch
// path is always a cache hit. It is deliberately *not* called from static
// init — the caller (typically the mlx-node / Device bringup path) decides
// when to invoke it so startup can overlap warmup with other work.
//
// The variant list is enumerated by hand rather than generated because it is
// small and we want the set to be greppable. Keep it in sync with the
// selection logic in `dispatch_matmul`:
//   - GEMV prefix:  `gemv_` for N < 16, `gemv_mc4_` for 16 ≤ N < 512,
//                   `gemv_mc8_` for N ≥ 512. Qwen3.5 hidden=896,
//                   intermediate=4864, vocab=151936 all ≥ 512 so mc8 is the
//                   dominant path; mc4 is cached to cover narrower
//                   projections and `gemv_` for the degenerate N < 16 case.
//   - `_al` suffix: fires when N % cols_per_wg == 0 (always true for
//                   Qwen3.5 dims, which are multiples of 8).
//   - `_bf16p`:     fires after the first dispatch flips the weight buffer
//                   into StorageMode::PackedBf16 — we pre-compile both the
//                   packed and non-packed variants so neither dispatch
//                   pays compile latency.
//   - `_sg<N>`:     appended by `wgpu::subgroup_suffix()` when the subgroup
//                   path is enabled; empty on hardware without subgroups.
//                   Handled automatically by the helper below.
void warmup_matmul_pipelines_impl(bool include_packed_bf16) {
  auto& dev = wgpu::device();
  const int sg = wgpu::effective_subgroup_size();
  const std::string sg_suffix = wgpu::subgroup_suffix();

  // Qwen3.5 weights are bf16, which the backend widens to f32 on upload
  // (dtype_to_wgsl maps bfloat16 → "f32"). f16 decode would produce a
  // separate variant set if shader-f16 is ever enabled, but Qwen3.5-0.8B
  // does not rely on it today, so keep the list tight.
  const char* wgsl_type = "f32";

  auto warm_gemv = [&](const char* prefix,
                       bool a_transposed,
                       bool b_transposed,
                       uint32_t cols_per_wg,
                       bool n_aligned,
                       bool b_packed_bf16) {
    std::string trans_suffix;
    trans_suffix += (a_transposed ? "T" : "N");
    trans_suffix += (b_transposed ? "T" : "N");
    std::string entry_name =
        std::string(prefix) + trans_suffix + "_" + wgsl_type;
    if (b_packed_bf16) {
      entry_name += "_bf16p";
    }
    entry_name += sg_suffix;
    if (n_aligned) {
      entry_name += "_al";
    }
    const bool is_multi_col = (std::string(prefix) != "gemv_");
    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name,
        [&]() {
          return is_multi_col
              ? make_gemv_multi_col_kernel(
                    entry_name,
                    wgsl_type,
                    a_transposed,
                    b_transposed,
                    sg,
                    cols_per_wg,
                    n_aligned,
                    b_packed_bf16)
              : make_gemv_kernel(
                    entry_name,
                    wgsl_type,
                    a_transposed,
                    b_transposed,
                    sg,
                    b_packed_bf16);
        });
    (void)dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());
  };

  auto warm_gemm = [&](bool a_transposed,
                       bool b_transposed,
                       bool b_packed_bf16) {
    std::string trans_suffix;
    trans_suffix += (a_transposed ? "T" : "N");
    trans_suffix += (b_transposed ? "T" : "N");
    std::string entry_name =
        std::string("gemm_") + trans_suffix + "_" + wgsl_type;
    if (b_packed_bf16) {
      entry_name += "_bf16p";
    }
    // GEMM does not take the subgroup path (dispatch_matmul passes sg=0 for
    // GEMM) and does not append sg_suffix. Match that here so cache keys
    // line up with the production dispatch.
    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name,
        [&]() {
          return make_gemm_kernel(
              entry_name, wgsl_type, a_transposed, b_transposed, b_packed_bf16);
        });
    (void)dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());
  };

  // ---- GEMV NT (Qwen decode hot path; b_transposed=true via
  // check_transpose() on every linear weight). ----
  warm_gemv("gemv_mc4_", false, true, 4u, /*n_aligned=*/true, false);
  warm_gemv("gemv_mc8_", false, true, 8u, /*n_aligned=*/true, false);
  warm_gemv("gemv_", false, true, 1u, /*n_aligned=*/false, false);

  // ---- GEMM NT at TILE_SIZE=16 (prefill / Tq > 1 hot path). ----
  warm_gemm(false, true, /*b_packed_bf16=*/false);

  if (!include_packed_bf16) {
    return;
  }
  // ---- Packed-bf16 siblings. Packed variants require b_transposed=true
  // (enforced by make_gemv_*/make_gemm_kernel assertions). ----
  warm_gemv("gemv_mc4_", false, true, 4u, /*n_aligned=*/true, true);
  warm_gemv("gemv_mc8_", false, true, 8u, /*n_aligned=*/true, true);
  warm_gemv("gemv_", false, true, 1u, /*n_aligned=*/false, true);
  warm_gemm(false, true, /*b_packed_bf16=*/true);
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

  // MM-F018: the epilogue path below binds `c_in` as read-only storage and
  // `out_buf` as read_write in the same compute pass. WebGPU forbids the same
  // buffer from appearing in both roles, so if a graph-level optimizer ever
  // donates C's buffer to `out` (C == out going in) the dispatch would fail
  // validation. Call `ensure_no_alias` *before* the matmul dispatch so that
  // any reallocation receives the matmul output and the epilogue then reads
  // C and writes out from distinct buffers. Matches the idiom used in
  // unary/binary/ternary/compiled primitives. In practice `out.set_data`
  // above already allocates a fresh buffer so this is defensive, but it
  // costs nothing and keeps the invariant explicit.
  wgpu::ensure_no_alias(out, c);

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

// ---------------------------------------------------------------------------
// MM-F014: free-function entry point for pipeline warmup
// ---------------------------------------------------------------------------
//
// Intentionally *not* auto-called from static init. The caller decides when
// to invoke it (typically once, right after device bringup in mlx-node's
// init path) so startup can overlap warmup with other work. Callers declare
// this forward in their own TU:
//
//   namespace mlx::core { void warmup_matmul_pipelines(); }
//
// until a shared header for WebGPU init hooks lands. That header lives
// outside this agent's scope.
//
// TODO(matmul/warmup): expose a public entry point — needs a header change
// (e.g. `mlx/backend/webgpu/warmup.h`) that is owned by the device/worker
// agent. When that lands, move this declaration into the header and drop
// the manual forward-declaration above.
void warmup_matmul_pipelines() {
  warmup_matmul_pipelines_impl(/*include_packed_bf16=*/true);
}

} // namespace mlx::core
