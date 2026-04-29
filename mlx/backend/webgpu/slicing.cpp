// Copyright 2026 Apple Inc.

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"

#include <limits>
#include <numeric>
#include <stdexcept>

namespace mlx::core {

// slice_gpu and pad_gpu are provided by gpu/slicing.cpp (identical logic).
// Only WebGPU-specific functions (concatenate_gpu, compute_dynamic_offset)
// are defined here to avoid duplicate symbol errors.

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  // Compute cumulative sizes along the concat axis.
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  auto strides = out.strides();

  // SPEC-F025: Detect when each input's slice of the output is itself
  // row-contiguous. This happens when the concat axis is effectively the
  // outermost non-trivial dim — i.e. all dims before `axis` have size 1.
  // In that case `out.strides()` restricted to the slice shape matches
  // make_contiguous_strides(inputs[i].shape()), so the slice occupies a
  // contiguous range in the output buffer. Downstream copy_gpu_inplace
  // can then take the Vector / same-dtype byte-copy fast path
  // (SPEC-F026) instead of the general strided kernel.
  bool slice_row_contig = true;
  for (int d = 0; d < axis; ++d) {
    if (out.shape(d) != 1) {
      slice_row_contig = false;
      break;
    }
  }

  // Flags for the slice view. For the row-contiguous case we can
  // advertise it so downstream fast paths can see through the view;
  // otherwise keep the pre-existing pessimistic flags (correct for
  // all layouts, just slower).
  auto slice_flags = out.flags();
  if (!slice_row_contig) {
    slice_flags.row_contiguous = false;
    slice_flags.col_contiguous = false;
    slice_flags.contiguous = false;
  }

  for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, slice_flags, out_slice.size(), data_offset);

    // SPEC-F026: KV-cache-append style concats (e.g. along seq axis with
    // outer dims of size 1) hit this shortcut: a contiguous input feeding
    // a contiguous dest slice is a straight byte copy, no need for the
    // GeneralGeneral strided-copy kernel. copy_gpu_inplace's Vector path
    // already has a copy_buffer_to_buffer fast path when dtypes match.
    const auto& in_i = inputs[i];
    bool vector_ok = slice_row_contig &&
        in_i.flags().row_contiguous && in_i.size() == in_i.data_size() &&
        in_i.dtype() == out.dtype();
    CopyType ctype = vector_ok ? CopyType::Vector : CopyType::GeneralGeneral;
    copy_gpu_inplace(in_i, out_slice, ctype, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  // For WebGPU, we use a CPU fallback to compute the dynamic offset.
  // This requires synchronizing to read index values from GPU memory.
  // This is acceptable because compute_dynamic_offset is called with
  // small index arrays (typically < 10 elements).

  auto& encoder = wgpu::get_command_encoder(s);

  encoder.set_input_array(indices);

  // Synchronize to ensure indices are available for CPU readback.
  encoder.synchronize();

  // Compute the offset on the CPU.
  int nidx = static_cast<int>(axes.size());
  int64_t acc = 0;

  switch (indices.dtype().val()) {
    case Dtype::Val::int32: {
      auto* ptr = indices.data<int32_t>();
      for (int i = 0; i < nidx; ++i) {
        acc += static_cast<int64_t>(ptr[i]) * strides[axes[i]];
      }
      break;
    }
    case Dtype::Val::int64: {
      auto* ptr = indices.data<int64_t>();
      for (int i = 0; i < nidx; ++i) {
        acc += ptr[i] * strides[axes[i]];
      }
      break;
    }
    case Dtype::Val::uint32: {
      auto* ptr = indices.data<uint32_t>();
      for (int i = 0; i < nidx; ++i) {
        acc += static_cast<int64_t>(ptr[i]) * strides[axes[i]];
      }
      break;
    }
    default: {
      auto* ptr = indices.data<int32_t>();
      for (int i = 0; i < nidx; ++i) {
        acc += static_cast<int64_t>(ptr[i]) * strides[axes[i]];
      }
      break;
    }
  }

  if (acc < 0 ||
      acc > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        "[WebGPU] dynamic slice offset exceeds WebGPU i32 range");
  }
  int32_t acc32 = static_cast<int32_t>(acc);

  // Return a single int32 offset value. WebGPU copy kernels index storage
  // buffers with 32-bit element offsets, and copy.cpp reads this scalar back
  // before folding it into the static copy offset.
  array offset({1}, int32, nullptr, {});
  offset.set_data(allocator::malloc(wgpu::wgpu_alloc_size(offset)));

  auto& dev = wgpu::device();
  WGPUBuffer dst_buf = wgpu::wgpu_buffer(offset);
  wgpuQueueWriteBuffer(
      dev.gpu_queue(), dst_buf, 0, &acc32, sizeof(acc32));

  return offset;
}

} // namespace mlx::core
