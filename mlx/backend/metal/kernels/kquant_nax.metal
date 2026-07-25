// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kquant_nax.h"

// Only qmm_t. `qmm` is the one dispatcher that reaches NAX with a plain
// (x, w, scales, biases) triple and a transposed weight, which is the prefill
// shape K-quants are slow on; the gather paths keep the simdgroup kernels, and
// nax_supports_mode in mlx/backend/metal/quantized.cpp admits K-quants for this
// path only.
//
// super_ratio is 256 / group_size, the number of sub-blocks a super-block's
// (d, dmin) covers; has_min says whether the sub-scales interleave a minimum.

#define instantiate_kquant_aligned_batched(mode, name, type, aligned, batched, group_size, bits, super_ratio, has_min, bm, bn, bk, wm, wn) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      kquant_ ## name, \
      type, \
      group_size, \
      bits, \
      super_ratio, \
      has_min, \
      aligned, \
      batched, \
      bm, \
      bk, \
      bn, \
      wm, \
      wn)

#define instantiate_kquant_nax_all(mode, type, group_size, bits, super_ratio, has_min) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, true, 1, group_size, bits, super_ratio, has_min, 64, 64, 64, 2, 2) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, true, 0, group_size, bits, super_ratio, has_min, 64, 64, 64, 2, 2) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, false, 1, group_size, bits, super_ratio, has_min, 64, 64, 64, 2, 2) \
  instantiate_kquant_aligned_batched(mode, qmm_t_nax, type, false, 0, group_size, bits, super_ratio, has_min, 64, 64, 64, 2, 2)

#define instantiate_kquant_nax_types(type) \
  instantiate_kquant_nax_all(q6k, type, 16, 6, 16, false) \
  instantiate_kquant_nax_all(q4k, type, 32, 4, 8, true) \
  instantiate_kquant_nax_all(q5k, type, 32, 5, 8, true)

instantiate_kquant_nax_types(float)
instantiate_kquant_nax_types(float16_t)
instantiate_kquant_nax_types(bfloat16_t)
    // clang-format on
