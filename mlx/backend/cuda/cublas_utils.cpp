// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/utils.h"

#include <cstring>

namespace mlx::core {

namespace cublas_utils {

cublasLtMatrixLayout_t create_matrix_layout(
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride) {
  cublasLtMatrixLayout_t desc;
  if (transposed) {
    std::swap(rows, cols);
  }
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&desc, type, rows, cols, ld));
  if (batch_count > 1) {
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(int32_t)));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_stride,
        sizeof(int64_t)));
  }
  return desc;
}

} // namespace cublas_utils

namespace {

auto& cublas_handles_cache() {
  struct CublasHandles {
    ~CublasHandles() {
      if (handle) {
        CHECK_CUBLAS_ERROR(cublasLtDestroy(handle));
        CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceDestroy(pref));
      }
    }
    cublasLtHandle_t handle{nullptr};
    cublasLtMatmulPreference_t pref{nullptr};
  };
  static thread_local std::vector<CublasHandles> cache(gpu::device_count());
  return cache;
}

auto get_cublas_handles(cu::Device& device) {
  auto& storage = cublas_handles_cache().at(device.cuda_device());
  if (!storage.handle) {
    // Create cublasLt handle.
    device.make_current();
    CHECK_CUBLAS_ERROR(cublasLtCreate(&storage.handle));
    // The recommended cublas workspace size is 4 MiB for pre-Hopper and 32
    // MiB for Hopper+:
    // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
    uint64_t MiB = 1024 * 1024;
    uint64_t workspace_size =
        device.compute_capability_major() >= 9 ? 32 * MiB : 4 * MiB;
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&storage.pref));
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceSetAttribute(
        storage.pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(uint64_t)));
  }
  return std::make_tuple(storage.handle, storage.pref);
}

struct CublasPlanEntry {
  cublasLtMatmulHeuristicResult_t heuristic{};
  std::string subgraph_key;
  bool is_updatable{true};
  bool has_heuristic{false};
  bool has_subgraph_key{false};
};

auto& cublas_plan_cache() {
  static thread_local LRUBytesKeyCache<
      CublasMatmulBase::PlanKey,
      CublasPlanEntry>
      cache("MLX_CUDA_CUBLAS_CACHE_SIZE", /* default_capacity */ 512);
  return cache;
}

inline uint8_t ptr_align_bucket(const void* ptr) {
  if (!ptr) {
    return 0;
  }
  auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  if ((addr & 15) == 0) {
    return 3;
  }
  if ((addr & 7) == 0) {
    return 2;
  }
  return 1;
}

} // namespace

void check_cublas_error(const char* name, cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    // TODO: Use cublasGetStatusString when it is widely available.
    throw std::runtime_error(
        fmt::format("{} failed with code: {}.", name, static_cast<int>(err)));
  }
}

void init_cublas_handles_cache() {
  cublas_handles_cache();
}

CublasMatmulBase::~CublasMatmulBase() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
}

void CublasMatmulBase::init_base(
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
    int64_t b_batch_stride) {
  M_ = a_rows;
  N_ = b_cols;
  scale_type_ = scale_type;
  std::tie(handle_, pref_) = get_cublas_handles(device);
  heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

  std::memset(&plan_key_, 0, sizeof(plan_key_));
  plan_key_.device = device.cuda_device();
  plan_key_.a_transposed = a_transposed ? 1 : 0;
  plan_key_.b_transposed = b_transposed ? 1 : 0;
  plan_key_.scale_type = static_cast<int32_t>(scale_type);
  plan_key_.compute_type = static_cast<int32_t>(compute_type);
  plan_key_.data_type = static_cast<int32_t>(data_type);
  plan_key_.output_type = static_cast<int32_t>(output_type);
  plan_key_.batch_count = batch_count;
  plan_key_.a_rows = a_rows;
  plan_key_.a_cols = a_cols;
  plan_key_.lda = lda;
  plan_key_.b_rows = b_rows;
  plan_key_.b_cols = b_cols;
  plan_key_.ldb = ldb;
  plan_key_.a_batch_stride = a_batch_stride;
  plan_key_.b_batch_stride = b_batch_stride;

  CHECK_CUBLAS_ERROR(
      cublasLtMatmulDescCreate(&matmul_desc_, compute_type, scale_type));

  // In cublasLt matrices use column-major layout, while it is possible to use
  // the CUBLASLT_ORDER_ROW option to switch to row-major layout, the bias
  // epilogue does not work with the option. So instead we swap A and B to make
  // cublasLt return the row-major result, which works because:
  // - the data of a matrix in row-major layout is identical to its transpose in
  //   column-major layout
  // - C^T = (A @ B)^T = B^T @ A^T
  cublasOperation_t a_op = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &a_op,
      sizeof(cublasOperation_t)));
  cublasOperation_t b_op = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &b_op,
      sizeof(cublasOperation_t)));

  a_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      b_cols,
      b_rows,
      b_transposed,
      ldb,
      batch_count,
      b_batch_stride);
  b_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      a_cols,
      a_rows,
      a_transposed,
      lda,
      batch_count,
      a_batch_stride);
  out_desc_ = cublas_utils::create_matrix_layout(
      output_type, b_cols, a_rows, false, b_cols, batch_count, b_cols * a_rows);
}

void CublasMatmulBase::execute_matmul(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr) {
  BytesKey<PlanKey> cache_key;
  std::memcpy(&cache_key.pod, &plan_key_, sizeof(PlanKey));
  cache_key.pod.has_c = c ? 1 : 0;
  cache_key.pod.align_bits = static_cast<uint8_t>(
      ptr_align_bucket(out) | (ptr_align_bucket(a) << 2) |
      (ptr_align_bucket(b) << 4) | (ptr_align_bucket(c) << 6));

  auto& cache = cublas_plan_cache();
  auto it = cache.find(cache_key);
  if (it == cache.end()) {
    it = cache.emplace(cache_key, CublasPlanEntry{}).first;
  }
  auto& entry = it->second;

  if (entry.has_heuristic) {
    heuristic_ = entry.heuristic;
  } else if (heuristic_.state != CUBLAS_STATUS_SUCCESS) {
    int ret = 0;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
        handle_,
        matmul_desc_,
        a_desc_,
        b_desc_,
        c ? c_desc_ : out_desc_,
        out_desc_,
        pref_,
        1,
        &heuristic_,
        &ret));
    if (ret == 0) {
      throw std::runtime_error("Can not find algorithm for matmul.");
    }
    entry.heuristic = heuristic_;
    entry.has_heuristic = true;
  }

  void* workspace_ptr = allocate_workspace(encoder, heuristic_.workspaceSize);

  // Execute matmul
  auto capture = encoder.capture_context();
  try {
    CHECK_CUBLAS_ERROR(cublasLtMatmul(
        handle_,
        matmul_desc_,
        alpha_ptr,
        b, // a and b are swapped for row-major layout
        a_desc_,
        a,
        b_desc_,
        beta_ptr,
        c ? c : out,
        c ? c_desc_ : out_desc_,
        out,
        out_desc_,
        &heuristic_.algo,
        workspace_ptr,
        heuristic_.workspaceSize,
        encoder.stream()));
  } catch (...) {
    capture.discard = true;
    throw;
  }
  capture.discard = true;
  encoder.add_captured_graph_cached(
      capture.graph,
      entry.subgraph_key,
      entry.is_updatable,
      entry.has_subgraph_key);
}

void CublasMatmulBase::set_bias(
    cu::CommandEncoder& encoder,
    const array& bias) {
  encoder.set_input_array(bias);
  plan_key_.has_bias = 1;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));
  auto* bias_ptr = gpu_ptr<void>(bias);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
}

} // namespace mlx::core
