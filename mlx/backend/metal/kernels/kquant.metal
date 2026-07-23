// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kquant.h"

// Only the three shapes ggml defines: q6k is a symmetric 6-bit sub-block of 16,
// q4k and q5k are asymmetric sub-blocks of 32. super_ratio is 256 / group_size,
// the number of sub-blocks a super-block's (d, dmin) covers.
//
// There is no quantize: K-quants are only ever consumed, and producing one
// needs ggml's make_qkx2_quants search. There is no qmv_quad either: dispatch_qmv
// only routes there when the reduction dim is 64 or 128, and the K-quant
// contract puts it at a multiple of 256.

#define instantiate_kquant(mode, name, type, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min)

#define instantiate_kquant_batched(mode, name, type, batched, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      batched)

#define instantiate_kquant_aligned(mode, name, type, aligned, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      aligned)

#define instantiate_kquant_aligned_batched(mode, name, type, aligned, batched, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      aligned, \
      batched)

#define instantiate_kquant_wide(mode, name, type, vecs_per_tg, k_lanes, batched, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      vecs_per_tg, \
      k_lanes, \
      batched)

#define instantiate_kquant_split_k(mode, name, type, split_k, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      split_k)

#define instantiate_kquant_rhs(mode, name, type, bm, bn, bk, wm, wn, transpose, group_size, bits, super_ratio, has_min) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      kquant_gather_qmm_rhs, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      bm, \
      bn, \
      bk, \
      wm, \
      wn, \
      transpose)

#define instantiate_kquant_batched_wrap(mode, name, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_batched(mode, name, type, 1, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_batched(mode, name, type, 0, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_all_batched(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_batched_wrap(mode, qmv_fast, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_batched_wrap(mode, qmv, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_batched_wrap(mode, qvm, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_batched_wrap(mode, qmm_n, type, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_all_single(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant(mode, dequantize, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant(mode, gather_qmv_fast, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant(mode, gather_qmv, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant(mode, gather_qvm, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant(mode, gather_qmm_n, type, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_all_aligned(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned(mode, gather_qmm_t, type, true, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned(mode, gather_qmm_t, type, false, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned(mode, qmm_t_splitk, type, true, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned(mode, qmm_t_splitk, type, false, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, true, 1, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, true, 0, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, false, 1, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned_batched(mode, qmm_t, type, false, 0, group_size, bits, super_ratio, has_min)

// vecs_per_tg (input-vector tile) 2..5, k_lanes 8 as the affine family uses.
#define instantiate_kquant_wide_wrap(mode, type, vecs_per_tg, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_wide(mode, qmv_wide, type, vecs_per_tg, 8, 0, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_wide(mode, qmv_wide, type, vecs_per_tg, 8, 1, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_all_wide(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_wide_wrap(mode, type, 2, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_wide_wrap(mode, type, 3, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_wide_wrap(mode, type, 4, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_wide_wrap(mode, type, 5, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_all_splitk(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_split_k(mode, qvm_split_k, type, 8, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_split_k(mode, qvm_split_k, type, 32, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_all_rhs(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_rhs(mode, gather_qmm_rhs_nt, type, 16, 32, 32, 1, 2, true, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_rhs(mode, gather_qmm_rhs_nn, type, 16, 32, 32, 1, 2, false, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_modes(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_all_single(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_all_batched(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_all_aligned(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_all_wide(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_all_splitk(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_all_rhs(mode, type, group_size, bits, super_ratio, has_min)

#define instantiate_kquant_types(type) \
  instantiate_kquant_modes(q6k, type, 16, 6, 16, false) \
  instantiate_kquant_modes(q4k, type, 32, 4, 8, true) \
  instantiate_kquant_modes(q5k, type, 32, 5, 8, true)

instantiate_kquant_types(float)
instantiate_kquant_types(float16_t)
instantiate_kquant_types(bfloat16_t)
    // clang-format on
