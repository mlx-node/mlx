// Copyright 2026 Apple Inc.
//
// Stub eval_cpu implementations for BLAS/LAPACK-dependent primitives.
// These are needed when MLX_NO_BLAS=ON so that the C++ vtables are still
// emitted (the key function for each class is eval_cpu, and without a
// definition the linker leaves the vtable undefined, causing
// call_indirect "function signature mismatch" traps in WASM).

#include "mlx/primitives.h"

namespace mlx::core {

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Matmul::eval_cpu] Not available (no BLAS).");
}

void AddMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[AddMM::eval_cpu] Not available (no BLAS).");
}

void Convolution::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Convolution::eval_cpu] Not available (no BLAS).");
}

void BlockMaskedMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[BlockMaskedMM::eval_cpu] Not available (no BLAS).");
}

void GatherMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[GatherMM::eval_cpu] Not available (no BLAS).");
}

void SegmentedMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[SegmentedMM::eval_cpu] Not available (no BLAS).");
}

void Eig::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[Eig::eval_cpu] Not available (no BLAS).");
}

void Eigh::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[Eigh::eval_cpu] Not available (no BLAS).");
}

void LUF::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[LUF::eval_cpu] Not available (no BLAS).");
}

void QRF::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[QRF::eval_cpu] Not available (no BLAS).");
}

void SVD::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[SVD::eval_cpu] Not available (no BLAS).");
}

void Inverse::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Inverse::eval_cpu] Not available (no BLAS).");
}

void Cholesky::eval_cpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Cholesky::eval_cpu] Not available (no BLAS).");
}

} // namespace mlx::core
