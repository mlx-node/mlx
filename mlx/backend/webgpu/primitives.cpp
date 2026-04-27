// Copyright © 2026 Apple Inc.

#include "mlx/primitives.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no WebGPU implementation."); \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no WebGPU implementation."); \
  }

namespace mlx::core {

// fast::ScaledDotProductAttention — use_fallback, supports_bool_mask, and
// eval_gpu are implemented in sdpa.cpp (fused vector kernel for decode).

bool fast::ScaledDotProductAttentionVJP::use_fallback(
    const array& q,
    Stream s) {
  return true;
}

// Binary ops — implemented in binary.cpp:
//   Add, Subtract, Multiply, Divide, Remainder, Power,
//   Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual,
//   LogicalAnd, LogicalOr, Maximum, Minimum, LogAddExp, ArcTan2

// Unary ops — implemented in unary.cpp:
//   Abs, Negative, Exp, Expm1, Log, Log1p, Sigmoid, Sign,
//   Sin, Cos, Tan, ArcSin, ArcCos, ArcTan,
//   Sinh, Cosh, Tanh, ArcSinh, ArcCosh, ArcTanh,
//   Sqrt, Square, Ceil, Floor, Round, Erf, ErfInv,
//   LogicalNot, BitwiseInvert

// AddMM, Matmul — implemented in matmul.cpp
// Arange — implemented in arange.cpp
// Gather, GatherAxis — implemented in indexing.cpp
// Scatter, ScatterAxis — implemented in indexing.cpp
// Select — implemented in ternary.cpp
// QuantizedMatmul, GatherQMM, fast::Quantize, QQMatmul — implemented in quantized.cpp
// Reduce — implemented in reduce.cpp
// Softmax — implemented in softmax.cpp

// Ops implemented by gpu/primitives.cpp (metadata-only or using generic GPU copy):
//   AsStrided, AsType, Broadcast, BroadcastAxes, Concatenate, Contiguous,
//   Copy, CustomTransforms, Depends, DynamicSlice, DynamicSliceUpdate,
//   ExpandDims, Flatten, Full, NumberOfElements, Pad, Reshape, Slice,
//   Split, Squeeze, StopGradient, Transpose, Unflatten, View

// ArgPartition — implemented in sort.cpp
// ArgReduce — implemented in arg_reduce.cpp
// ArgSort — implemented in sort.cpp
NO_GPU(BitwiseBinary)
NO_GPU(BlockMaskedMM)
// Compiled — implemented in compiled.cpp
NO_GPU(Conjugate)
// Convolution — implemented in conv.cpp
NO_GPU_MULTI(DivMod)
NO_GPU(FFT)
NO_GPU(GatherMM)
NO_GPU(Hadamard)
NO_GPU(Imag)
NO_GPU(Load)
// LogSumExp — implemented in logsumexp.cpp
NO_GPU_MULTI(LUF)
NO_GPU(Partition)
NO_GPU_MULTI(QRF)
// RandomBits — implemented in random.cpp
NO_GPU(Real)
// Scan — implemented in scan.cpp
NO_GPU(SegmentedMM)
// SliceUpdate — implemented in indexing.cpp
// Sort — implemented in sort.cpp
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eigh)
NO_GPU_MULTI(Eig)
NO_GPU(MaskedScatter)

namespace fast {
// LayerNorm — implemented in normalization.cpp
NO_GPU_MULTI(LayerNormVJP)
// RMSNorm — implemented in normalization.cpp
NO_GPU_MULTI(RMSNormVJP)
// RoPE — implemented in rope.cpp
// ScaledDotProductAttention — implemented in sdpa.cpp
NO_GPU_MULTI(ScaledDotProductAttentionVJP)
NO_GPU_MULTI(ConvertFP8)
NO_GPU_MULTI(CustomKernel)
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core
