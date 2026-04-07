// Copyright 2026 Apple Inc.

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"

#include <cstring>
#include <numeric>

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

  out.set_data(allocator::malloc(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
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

  // Prepare output array for a single int64 offset value.
  array offset({1}, int64, nullptr, {});
  offset.set_data(allocator::malloc(offset.itemsize()));
  encoder.add_temporary(offset);
  encoder.set_input_array(indices);
  encoder.set_output_array(offset);

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

  // Write the computed offset into the GPU buffer via a staging buffer.
  auto& dev = wgpu::device();

  WGPUBufferDescriptor buf_desc = {};
  buf_desc.size = sizeof(int64_t);
  buf_desc.usage = WGPUBufferUsage_CopySrc;
  buf_desc.mappedAtCreation = true;

  WGPUBuffer staging = wgpuDeviceCreateBuffer(dev.gpu_device(), &buf_desc);
  if (staging) {
    void* mapped = wgpuBufferGetMappedRange(staging, 0, sizeof(int64_t));
    std::memcpy(mapped, &acc, sizeof(int64_t));
    wgpuBufferUnmap(staging);

    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder cmd_enc =
        wgpuDeviceCreateCommandEncoder(dev.gpu_device(), &enc_desc);
    WGPUBuffer dst_buf = wgpu::wgpu_buffer(offset);
    wgpuCommandEncoderCopyBufferToBuffer(
        cmd_enc, staging, 0, dst_buf, 0, sizeof(int64_t));
    WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, nullptr);
    wgpuQueueSubmit(dev.gpu_queue(), 1, &cmd_buf);
    wgpuCommandBufferRelease(cmd_buf);
    wgpuCommandEncoderRelease(cmd_enc);
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
  }

  return offset;
}

} // namespace mlx::core
