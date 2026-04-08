// Copyright 2026 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "gen/wgsl_sources.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// Must match the WGSL CopyParams struct layout with vec4 packing.
//
// WGSL layout (uniform address space, vec4 aligned):
//   Offset  0: size_ndim_offsets (vec4<u32>) -- .x=size, .y=ndim,
//                                               .z=in_offset, .w=out_offset
//   Offset 16: shape_0        (vec4<u32>)  -- shape[0..3]
//   Offset 32: shape_1        (vec4<u32>)  -- shape[4..7]
//   Offset 48: in_strides_0   (vec4<i32>)  -- in_strides[0..3]
//   Offset 64: in_strides_1   (vec4<i32>)  -- in_strides[4..7]
//   Offset 80: out_strides_0  (vec4<i32>)  -- out_strides[0..3]
//   Offset 96: out_strides_1  (vec4<i32>)  -- out_strides[4..7]
//   Total: 112 bytes
struct CopyParams {
  uint32_t size_ndim_offsets[4]; // [size, ndim, in_offset, out_offset]
  uint32_t shape_0[4];          // shape[0..3]
  uint32_t shape_1[4];          // shape[4..7]
  int32_t in_strides_0[4];      // in_strides[0..3]
  int32_t in_strides_1[4];      // in_strides[4..7]
  int32_t out_strides_0[4];     // out_strides[0..3]
  int32_t out_strides_1[4];     // out_strides[4..7]
};

// Get or create the shader module for copy kernels.
WGPUShaderModule get_copy_shader_module() {
  auto& dev = wgpu::device();
  return dev.get_or_create_shader_module(
      "copy_kernels",
      []() { return std::string(wgpu::kernels::copy()); });
}

// Map CopyType to the WGSL entry point name.
const char* copy_type_to_entry_point(CopyType ctype) {
  switch (ctype) {
    case CopyType::Scalar:
      return "copy_s";
    case CopyType::Vector:
      return "copy_v";
    case CopyType::General:
      return "copy_g";
    case CopyType::GeneralGeneral:
      return "copy_gg";
  }
  throw std::runtime_error("[WebGPU] Unhandled CopyType");
}

// Dispatch a copy kernel.
void dispatch_copy(
    wgpu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    const array& out,
    uint32_t elem_count,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out) {
  auto& dev = wgpu::device();
  WGPUShaderModule shader = get_copy_shader_module();

  const char* entry_point = copy_type_to_entry_point(ctype);
  std::string pipeline_key = std::string("copy_") + entry_point;

  auto pe =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_point);

  // Compute the effective element offsets into array<u32> buffers.
  // arr.offset() returns a byte offset from the buffer base pointer.
  // The shader indexes array<u32>, so we need u32 element indices.
  // On GPU, all types occupy one u32 slot (bf16→f32, bool→u32), so
  // element offset = u32 index. Convert byte offset → element offset
  // by dividing by CPU itemsize, then add the parameter element offset.
  uint32_t in_elem =
      static_cast<uint32_t>(in.offset()) / static_cast<uint32_t>(in.itemsize())
      + static_cast<uint32_t>(in_offset);
  uint32_t out_elem =
      static_cast<uint32_t>(out.offset()) / static_cast<uint32_t>(out.itemsize())
      + static_cast<uint32_t>(out_offset);
  // Each element maps to one u32 on GPU (wgpu_itemsize is always 4 currently).
  uint32_t effective_in_offset = in_elem;
  uint32_t effective_out_offset = out_elem;

  // Fill the uniform buffer. For scalar/vector kernels, only size and offsets
  // matter, but we always upload the full CopyParams struct.
  CopyParams params{};
  params.size_ndim_offsets[0] = elem_count;
  params.size_ndim_offsets[1] = static_cast<uint32_t>(shape.size());
  params.size_ndim_offsets[2] = effective_in_offset;
  params.size_ndim_offsets[3] = effective_out_offset;

  // Compute type conversion mode for cross-type copies
  uint32_t conversion_mode = 0;
  if (in.dtype() != out.dtype()) {
    auto in_val = in.dtype().val();
    auto out_val = out.dtype().val();
    if (in_val == Dtype::Val::bool_) {
      // bool (u32 0/1) → float types: 0→0.0, 1→1.0
      if (out_val == Dtype::Val::float32 || out_val == Dtype::Val::float16 ||
          out_val == Dtype::Val::bfloat16) {
        conversion_mode = 1;
      }
    } else if (out_val == Dtype::Val::bool_) {
      // any type → bool: nonzero → 1, zero → 0
      conversion_mode = 2;
    } else if (in_val == Dtype::Val::int32 &&
               (out_val == Dtype::Val::float32 || out_val == Dtype::Val::float16 ||
                out_val == Dtype::Val::bfloat16)) {
      conversion_mode = 3;
    } else if ((in_val == Dtype::Val::float32 || in_val == Dtype::Val::float16 ||
                in_val == Dtype::Val::bfloat16) &&
               out_val == Dtype::Val::int32) {
      conversion_mode = 4;
    } else if (in_val == Dtype::Val::uint32 &&
               (out_val == Dtype::Val::float32 || out_val == Dtype::Val::float16 ||
                out_val == Dtype::Val::bfloat16)) {
      conversion_mode = 5;
    }
  }
  // Pack mode into upper 16 bits of ndim field
  params.size_ndim_offsets[1] |= (conversion_mode << 16);

  uint32_t ndim = params.size_ndim_offsets[1] & 0xFFFFu;
  for (uint32_t i = 0; i < ndim && i < wgpu::MAX_NDIM; ++i) {
    if (i < 4) {
      params.shape_0[i] = static_cast<uint32_t>(shape[i]);
      params.in_strides_0[i] = static_cast<int32_t>(strides_in[i]);
      params.out_strides_0[i] = static_cast<int32_t>(strides_out[i]);
    } else {
      params.shape_1[i - 4] = static_cast<uint32_t>(shape[i]);
      params.in_strides_1[i - 4] = static_cast<int32_t>(strides_in[i]);
      params.out_strides_1[i - 4] = static_cast<int32_t>(strides_out[i]);
    }
  }

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(CopyParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(CopyParams)}});

  // Compute workgroup count:
  // Each thread processes N_READS elements; WORKGROUP_SIZE threads per group.
  uint32_t total_threads =
      (elem_count + wgpu::N_READS - 1) / wgpu::N_READS;
  uint32_t num_workgroups =
      (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;

  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace

// ============================================================================
// Public API implementation
// ============================================================================

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(in, out, ctype);
  if (donated && in.dtype() == out.dtype()) {
    return;
  }
  // WebGPU: prevent buffer aliasing between input and output bindings.
  // This can happen when set_copy_output_data donates the input buffer to the
  // output (e.g. AsType with same-sized types). The copy dispatch would then
  // bind the same buffer as both read-only and read-write, which WebGPU forbids.
  wgpu::ensure_no_alias(out, in);
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  if (out.size() == 0) {
    return;
  }

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  if (ctype == CopyType::Scalar || ctype == CopyType::Vector) {
    // For array<u32> buffers, element count = total bytes / 4.
    // For 4-byte types this equals data_size; for other sizes we adjust.
    uint32_t byte_count =
        static_cast<uint32_t>(out.data_size()) *
        static_cast<uint32_t>(wgpu::wgpu_itemsize(out.dtype()));
    uint32_t elem_count = (byte_count + 3u) / 4u;
    dispatch_copy(
        encoder, ctype, in, out, elem_count, i_offset, o_offset, {}, {}, {});
    return;
  }

  // General or GeneralGeneral: collapse contiguous dims for efficiency.
  // Note: strides are in units of the dtype's element size. Since we index
  // array<u32> in the shader, this is correct when itemsize == 4 (the common
  // case for WebGPU which maps all types to 32-bit equivalents). For sub-32-bit
  // types, strides would need scaling by itemsize/4.
  auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
      data_shape,
      std::vector<Strides>{strides_in, strides_out},
      INT32_MAX);

  uint32_t total_size = 1;
  for (auto& dim : shape_collapsed) {
    total_size *= static_cast<uint32_t>(dim);
  }

  dispatch_copy(
      encoder,
      ctype,
      in,
      out,
      total_size,
      i_offset,
      o_offset,
      shape_collapsed,
      strides_vec[0],
      strides_vec[1]);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(val);
  encoder.set_output_array(out);

  uint32_t byte_count =
      static_cast<uint32_t>(out.data_size()) *
      static_cast<uint32_t>(wgpu::wgpu_itemsize(out.dtype()));
  uint32_t elem_count = (byte_count + 3u) / 4u;
  dispatch_copy(
      encoder, CopyType::Scalar, val, out, elem_count, 0, 0, {}, {}, {});
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

} // namespace mlx::core
