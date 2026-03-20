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

bool fast::ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  return true;
}

bool fast::ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

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

// AddMM — implemented in matmul.cpp
NO_GPU(Arange)
NO_GPU(ArgPartition)
NO_GPU(ArgReduce)
NO_GPU(ArgSort)
NO_GPU(AsType)
NO_GPU(AsStrided)
NO_GPU(BitwiseBinary)
NO_GPU(BlockMaskedMM)
NO_GPU(Broadcast)
NO_GPU(BroadcastAxes)
NO_GPU_MULTI(Compiled)
NO_GPU(Concatenate)
NO_GPU(Conjugate)
NO_GPU(Contiguous)
NO_GPU(Convolution)
NO_GPU(Copy)
NO_GPU_MULTI(CustomTransforms)
NO_GPU_MULTI(Depends)
NO_GPU_MULTI(DivMod)
NO_GPU(DynamicSlice)
NO_GPU(DynamicSliceUpdate)
NO_GPU(NumberOfElements)
NO_GPU(ExpandDims)
NO_GPU(FFT)
NO_GPU(Flatten)
NO_GPU(Full)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU(Imag)
NO_GPU(Load)
NO_GPU(LogSumExp)
NO_GPU_MULTI(LUF)
// Matmul — implemented in matmul.cpp
NO_GPU(Pad)
NO_GPU(Partition)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(QQMatmul)
NO_GPU(RandomBits)
NO_GPU(Real)
// Reduce — implemented in reduce.cpp
NO_GPU(Reshape)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(SegmentedMM)
NO_GPU(Slice)
NO_GPU(SliceUpdate)
// Softmax — implemented in softmax.cpp
NO_GPU(Sort)
NO_GPU_MULTI(Split)
NO_GPU(Squeeze)
NO_GPU(StopGradient)
NO_GPU_MULTI(SVD)
NO_GPU(Transpose)
NO_GPU(Unflatten)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eigh)
NO_GPU_MULTI(Eig)
NO_GPU(View)
NO_GPU(MaskedScatter)

namespace fast {
NO_GPU_USE_FALLBACK(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_USE_FALLBACK(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_USE_FALLBACK(RoPE)
NO_GPU_MULTI(ScaledDotProductAttention)
NO_GPU_MULTI(ScaledDotProductAttentionVJP)
NO_GPU_MULTI(ConvertFP8)
NO_GPU_MULTI(Quantize)
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
