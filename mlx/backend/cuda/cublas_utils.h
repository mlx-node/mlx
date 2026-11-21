// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"

#include <cublasLt.h>

namespace mlx::core {
namespace cublas_utils {

cublasLtMatrixLayout_t create_matrix_layout(
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride);

inline cudaDataType_t dtype_to_cublas_type(Dtype dtype, std::string_view tag) {
  switch (dtype) {
    case float16:
      return CUDA_R_16F;
    case bfloat16:
      return CUDA_R_16BF;
    case float32:
      return CUDA_R_32F;
    case float64:
      return CUDA_R_64F;
    case complex64:
      return CUDA_C_32F;
    default:
      throw std::runtime_error(
          fmt::format(
              "Unsupported dtype in {}: {}.", tag, dtype_to_string(dtype)));
  }
}

} // namespace cublas_utils

void check_cublas_error(const char* name, cublasStatus_t err);

#define CHECK_CUBLAS_ERROR(cmd) check_cublas_error(#cmd, (cmd))

void init_cublas_handles_cache();

class CublasMatmulBase {
 public:
  struct PlanKey {
    int device{0};
    uint8_t plan_kind{0};
    uint8_t a_transposed{0};
    uint8_t b_transposed{0};
    uint8_t has_bias{0};
    uint8_t has_c{0};
    uint8_t has_out_override{0};
    uint8_t out_transposed{0};
    uint8_t align_bits{0};
    int32_t scale_type{0};
    int32_t compute_type{0};
    int32_t data_type{0};
    int32_t output_type{0};
    int32_t a_scale_mode{0};
    int32_t b_scale_mode{0};
    int32_t batch_count{0};
    int32_t out_batch_count{0};
    uint64_t a_rows{0};
    uint64_t a_cols{0};
    int64_t lda{0};
    uint64_t b_rows{0};
    uint64_t b_cols{0};
    int64_t ldb{0};
    int64_t ldc{0};
    int64_t a_batch_stride{0};
    int64_t b_batch_stride{0};
    int64_t c_batch_stride{0};
    uint64_t out_rows{0};
    uint64_t out_cols{0};
    int64_t out_ld{0};
    int64_t out_batch_stride{0};
  };

  virtual ~CublasMatmulBase();

  void set_bias(cu::CommandEncoder& encoder, const array& bias);

 protected:
  CublasMatmulBase() = default;

  // Common member variables shared by all matmul types
  uint64_t M_;
  uint64_t N_;
  cudaDataType_t scale_type_;
  cublasLtMatmulPreference_t pref_{nullptr};
  cublasLtHandle_t handle_{nullptr};
  cublasLtMatmulDesc_t matmul_desc_{nullptr};
  cublasLtMatrixLayout_t a_desc_{nullptr};
  cublasLtMatrixLayout_t b_desc_{nullptr};
  cublasLtMatrixLayout_t c_desc_{nullptr};
  cublasLtMatrixLayout_t out_desc_{nullptr};
  cublasLtMatmulHeuristicResult_t heuristic_;
  PlanKey plan_key_{};

  void init_base(
      cu::Device& device,
      cudaDataType_t scale_type,
      cublasComputeType_t compute_type,
      cudaDataType_t data_type,
      cudaDataType_t output_type,
      bool a_transposed,
      uint64_t a_rows,
      uint64_t a_cols,
      int64_t lda,
      bool b_transposed,
      uint64_t b_rows,
      uint64_t b_cols,
      int64_t ldb,
      int32_t batch_count,
      int64_t a_batch_stride,
      int64_t b_batch_stride);

  void execute_matmul(
      cu::CommandEncoder& encoder,
      void* out,
      const void* a,
      const void* b,
      const void* c,
      const void* alpha_ptr,
      const void* beta_ptr);
};

} // namespace mlx::core
