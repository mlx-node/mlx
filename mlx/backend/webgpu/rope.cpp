// Copyright 2026 Apple Inc.
//
// WebGPU RoPE (Rotary Position Embedding) implementation.
//
// Two kernel variants:
//   1. "single" — T=1, contiguous, single offset (common inference path)
//      Dispatch: (dims/2, B*N, 1)
//   2. "general" — arbitrary T, offset per batch
//      Dispatch: (dims/2, T, B*N)
//
// Each thread handles one pair of elements and applies the rotation.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/fast_primitives.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mlx::core::fast {

namespace {

// ---------------------------------------------------------------------------
// RoPE uniform params — must match the WGSL struct layout (vec4-aligned)
// ---------------------------------------------------------------------------

// Single kernel: T=1, offset read from storage buffer
struct RoPESingleParams {
  uint32_t data[4]; // [dims_half, D, n_rows, pad]
  uint32_t data2[4]; // [scale_bits, base_log2_bits, traditional, forward]
  uint32_t offsets[4]; // [input_offset, output_offset, offset_offset, pad]
};

// General kernel: T>1, offset read from storage buffer
struct RoPEGeneralParams {
  uint32_t data[4]; // [dims_half, D, N, B]
  uint32_t data2[4]; // [scale_bits, base_log2_bits, traditional, forward]
  uint32_t data3[4]; // [T, offset_stride, mat_size, 0]
  uint32_t data4[4]; // [input_offset, output_offset, offset_offset, 0]
};

uint32_t shader_elem_offset(const array& arr, const char* op_name) {
  uint64_t offset =
      static_cast<uint64_t>(arr.offset()) / static_cast<uint64_t>(arr.itemsize());
  if (offset > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string(op_name) + " array offset exceeds WebGPU u32 indexing");
  }
  return static_cast<uint32_t>(offset);
}

struct RopeOffsetStorage {
  std::string tag;
  std::string wgsl_type;
  uint32_t offset_base;
  uint32_t stride_scale;
};

RopeOffsetStorage rope_offset_storage(const array& offset) {
  switch (offset.dtype().val()) {
    case Dtype::Val::int32:
      return {
          "i32",
          "i32",
          shader_elem_offset(offset, "[RoPE::eval_gpu] offset"),
          1u};
    case Dtype::Val::uint32:
      return {
          "u32",
          "u32",
          shader_elem_offset(offset, "[RoPE::eval_gpu] offset"),
          1u};
    case Dtype::Val::int64: {
      return {
          "i64",
          "i32",
          shader_elem_offset(offset, "[RoPE::eval_gpu] offset"),
          1u};
    }
    default:
      throw std::runtime_error(
          "[RoPE::eval_gpu] Unsupported offset dtype. Expected int32, "
          "uint32, or int64.");
  }
}

std::string rope_offset_expr(
    const RopeOffsetStorage& storage,
    const std::string& index) {
  if (storage.tag == "i32" || storage.tag == "u32") {
    return "f32(offsets[" + index + "])";
  }
  // WGSL has no i64 scalar type. The WebGPU upload path narrows int64 index
  // tensors to one i32 lane per logical element before kernels read them.
  return "f32(offsets[" + index + "])";
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: rope_single
// ---------------------------------------------------------------------------

std::string make_rope_single_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const RopeOffsetStorage& offset_storage) {
  std::ostringstream s;

  if (in_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";
  s << wgpu::wgsl_linear_thread_id();

  // Params struct
  s << "struct RoPEParams {\n"
    << "  data: vec4<u32>,\n"
    << "  data2: vec4<u32>,\n"
    << "  offsets: vec4<u32>,\n"
    << "}\n\n";

  // Bindings: input, output, offset, uniform
  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << in_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read> offsets: array<"
    << offset_storage.wgsl_type << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: RoPEParams;\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let dims_half = params.data.x;\n"
    << "  let D = params.data.y;\n"
    << "  let n_rows = params.data.z;\n"
    << "  let input_offset = params.offsets.x;\n"
    << "  let output_offset = params.offsets.y;\n"
    << "  let offset_index = params.offsets.z;\n"
    << "  let offset_val = " << rope_offset_expr(offset_storage, "offset_index") << ";\n"
    << "  let scale = bitcast<f32>(params.data2.x);\n"
    << "  let base_log2 = bitcast<f32>(params.data2.y);\n"
    << "  let traditional = params.data2.z;\n"
    << "  let fwd = params.data2.w;\n"
    << "\n"
    << "  // Each thread handles one pair in one row\n"
    << "  // Linearize gid into (pair_idx, row)\n"
    << "  let thread_id = linear_thread_id(wg_id, lid, nwg);\n"
    << "  let pair_idx = thread_id % dims_half;\n"
    << "  let row = thread_id / dims_half;\n"
    << "\n"
    << "  if (pair_idx >= dims_half || row >= n_rows) { return; }\n"
    << "\n"
    << "  let base_idx = row * D;\n"
    << "\n"
    << "  // Compute frequency\n"
    << "  let d = f32(pair_idx) / f32(dims_half);\n"
    << "  let inv_freq = exp2(-d * base_log2);\n"
    << "  let theta = scale * offset_val * inv_freq;\n"
    << "  let cos_theta = cos(theta);\n"
    << "  let sin_theta = sin(theta);\n"
    << "\n"
    << "  // Get pair indices\n"
    << "  var idx1: u32;\n"
    << "  var idx2: u32;\n"
    << "  if (traditional != 0u) {\n"
    << "    idx1 = base_idx + 2u * pair_idx;\n"
    << "    idx2 = idx1 + 1u;\n"
    << "  } else {\n"
    << "    idx1 = base_idx + pair_idx;\n"
    << "    idx2 = base_idx + pair_idx + dims_half;\n"
    << "  }\n"
    << "\n"
    << "  let x1 = f32(input[input_offset + idx1]);\n"
    << "  let x2 = f32(input[input_offset + idx2]);\n"
    << "\n"
    << "  if (fwd != 0u) {\n"
    << "    output[output_offset + idx1] = " << in_type << "(x1 * cos_theta - x2 * sin_theta);\n"
    << "    output[output_offset + idx2] = " << in_type << "(x1 * sin_theta + x2 * cos_theta);\n"
    << "  } else {\n"
    << "    output[output_offset + idx1] = " << in_type << "(x2 * sin_theta + x1 * cos_theta);\n"
    << "    output[output_offset + idx2] = " << in_type << "(x2 * cos_theta - x1 * sin_theta);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: rope_general
// ---------------------------------------------------------------------------

std::string make_rope_general_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const RopeOffsetStorage& offset_storage) {
  std::ostringstream s;

  if (in_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";
  s << wgpu::wgsl_linear_thread_id();

  // Params struct
  s << "struct RoPEParams {\n"
    << "  data: vec4<u32>,\n"
    << "  data2: vec4<u32>,\n"
    << "  data3: vec4<u32>,\n"
    << "  data4: vec4<u32>,\n"
    << "}\n\n";

  // Bindings: input, output, offset (storage), uniform
  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << in_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read> offsets: array<"
    << offset_storage.wgsl_type << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: RoPEParams;\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let dims_half = params.data.x;\n"
    << "  let D = params.data.y;\n"
    << "  let N = params.data.z;\n"
    << "  let B = params.data.w;\n"
    << "  let scale = bitcast<f32>(params.data2.x);\n"
    << "  let base_log2 = bitcast<f32>(params.data2.y);\n"
    << "  let traditional = params.data2.z;\n"
    << "  let fwd = params.data2.w;\n"
    << "  let T = params.data3.x;\n"
    << "  let offset_stride = params.data3.y;\n"
    << "  let mat_size = params.data3.z;\n"
    << "  let input_offset = params.data4.x;\n"
    << "  let output_offset = params.data4.y;\n"
    << "  let offset_base = params.data4.z;\n"
    << "\n"
    << "  // Linearize thread_id into (pair_idx, t, head_batch)\n"
    << "  let thread_id = linear_thread_id(wg_id, lid, nwg);\n"
    << "  let pair_idx = thread_id % dims_half;\n"
    << "  let remainder = thread_id / dims_half;\n"
    << "  let t = remainder % T;\n"
    << "  let head_batch = remainder / T;\n"
    << "\n"
    << "  let total_rows = B * N;\n"
    << "  if (pair_idx >= dims_half || t >= T || head_batch >= total_rows) { return; }\n"
    << "\n"
    << "  // Compute batch index for offset lookup\n"
    << "  let batch_idx = head_batch / N;\n"
    << "\n"
    << "  // Read offset for this batch\n"
    << "  let batch_offset_index = offset_base + batch_idx * offset_stride;\n"
    << "  let batch_offset = " << rope_offset_expr(offset_storage, "batch_offset_index") << ";\n"
    << "  let L = scale * (f32(t) + batch_offset);\n"
    << "\n"
    << "  // Compute frequency\n"
    << "  let d = f32(pair_idx) / f32(dims_half);\n"
    << "  let inv_freq = exp2(-d * base_log2);\n"
    << "  let theta = L * inv_freq;\n"
    << "  let cos_theta = cos(theta);\n"
    << "  let sin_theta = sin(theta);\n"
    << "\n"
    << "  // Compute base index: row = head_batch, position = t\n"
    << "  let base_idx = head_batch * mat_size + t * D;\n"
    << "\n"
    << "  // Get pair indices\n"
    << "  var idx1: u32;\n"
    << "  var idx2: u32;\n"
    << "  if (traditional != 0u) {\n"
    << "    idx1 = base_idx + 2u * pair_idx;\n"
    << "    idx2 = idx1 + 1u;\n"
    << "  } else {\n"
    << "    idx1 = base_idx + pair_idx;\n"
    << "    idx2 = base_idx + pair_idx + dims_half;\n"
    << "  }\n"
    << "\n"
    << "  let x1 = f32(input[input_offset + idx1]);\n"
    << "  let x2 = f32(input[input_offset + idx2]);\n"
    << "\n"
    << "  if (fwd != 0u) {\n"
    << "    output[output_offset + idx1] = " << in_type << "(x1 * cos_theta - x2 * sin_theta);\n"
    << "    output[output_offset + idx2] = " << in_type << "(x1 * sin_theta + x2 * cos_theta);\n"
    << "  } else {\n"
    << "    output[output_offset + idx1] = " << in_type << "(x2 * sin_theta + x1 * cos_theta);\n"
    << "    output[output_offset + idx2] = " << in_type << "(x2 * cos_theta - x1 * sin_theta);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point: fast::RoPE::eval_gpu
// ---------------------------------------------------------------------------

bool RoPE::use_fallback(Stream s) {
  return false;  // GPU implementation
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& offset = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();

  int ndim = in.ndim();
  int D = in.shape(-1);
  int T = in.shape(-2);
  int N = 1;
  for (int i = 1; i < ndim - 2; ++i) {
    N *= in.shape(i);
  }
  int B = in.shape(0);
  size_t mat_size = static_cast<size_t>(T) * D;

  // Handle partial rotation (dims < D): first copy entire input to output
  bool partial = dims_ < D;
  if (partial) {
    auto ctype =
        in.flags().row_contiguous ? CopyType::Vector : CopyType::General;
    copy_gpu(in, out, ctype, s);
  } else if (in.flags().row_contiguous) {
    out.set_data(
        allocator::malloc(wgpu::wgpu_alloc_size(out)),
        in.data_size(),
        in.strides(),
        in.flags());
  } else {
    // Copy non-contiguous input into output
    copy_gpu(in, out, CopyType::General, s);
  }

  uint32_t dims_half = static_cast<uint32_t>(dims_) / 2;
  if (dims_half == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string in_type(in_wgsl);
  auto offset_storage = rope_offset_storage(offset);

  // Determine if we can use the optimized single-offset path
  bool single = in.flags().row_contiguous && T == 1 && offset.size() == 1;

  // Pre-compute base_log2 = log2(base)
  float base_log2 = std::log2(base_);
  uint32_t scale_bits, base_log2_bits;
  std::memcpy(&scale_bits, &scale_, sizeof(scale_bits));
  std::memcpy(&base_log2_bits, &base_log2, sizeof(base_log2_bits));

  if (single) {
    // ----- Single offset path (T=1 inference) -----
    std::string entry_name =
        std::string("rope_single_") + in_type + "_" + offset_storage.tag;
    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name,
        [&]() {
          return make_rope_single_kernel(entry_name, in_type, offset_storage);
        });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    uint32_t n_rows = static_cast<uint32_t>(B * N);

    RoPESingleParams params{};
    params.data[0] = dims_half;
    params.data[1] = static_cast<uint32_t>(D);
    params.data[2] = n_rows;
    params.data[3] = 0;
    params.data2[0] = scale_bits;
    params.data2[1] = base_log2_bits;
    params.data2[2] = traditional_ ? 1u : 0u;
    params.data2[3] = forward_ ? 1u : 0u;
    params.offsets[0] = shader_elem_offset(in, "[RoPE::eval_gpu] input");
    params.offsets[1] = shader_elem_offset(out, "[RoPE::eval_gpu] output");
    params.offsets[2] = offset_storage.offset_base;
    params.offsets[3] = 0;

    // Always read from original input (not output) to avoid WebGPU
    // buffer aliasing — the same buffer can't be bound as both read-only
    // and read-write. For partial rotation, copy_gpu already put non-rotated
    // dims into output; we read rotated dims from input.
    encoder.set_input_array(in);
    encoder.set_input_array(offset);
    encoder.set_output_array(out);

    auto& pool = wgpu::device().uniform_pool();
    WGPUBuffer uniform_buf = pool.acquire(
        wgpu::device().gpu_queue(), &params, sizeof(RoPESingleParams));

    WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
    WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
    WGPUBuffer offset_buf = wgpu::wgpu_buffer(offset);
    uint64_t in_buf_size = wgpu::wgpu_bind_size(in);
    uint64_t out_buf_size = wgpu::wgpu_bind_size(out);
    uint64_t offset_buf_size = wgpu::wgpu_bind_size(offset);

    WGPUBindGroup bg = wgpu::create_bind_group(
        pe.layout,
        {{in_buf, in_buf_size},
         {out_buf, out_buf_size},
         {offset_buf, offset_buf_size},
         {uniform_buf, sizeof(RoPESingleParams)}});

    // Total threads needed: dims_half * n_rows
    // Dispatch as 1D with enough workgroups
    uint32_t total_threads = dims_half * n_rows;
    uint32_t n_workgroups =
        (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
    auto [wg_x, wg_y] =
        wgpu::get_2d_grid(n_workgroups, "[WebGPU rope_single]");
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);

    wgpuBindGroupRelease(bg);
    encoder.add_completed_handler([uniform_buf]() {
      wgpu::device().uniform_pool().release(uniform_buf);
    });

  } else {
    // ----- General path (T>1 or non-contiguous) -----
    std::string entry_name =
        std::string("rope_general_") + in_type + "_" + offset_storage.tag;
    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name,
        [&]() {
          return make_rope_general_kernel(entry_name, in_type, offset_storage);
        });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    // If we did a partial copy or general copy, the data is in the output.
    // Otherwise we read from input.
    bool read_from_out = partial || !in.flags().row_contiguous;

    uint32_t offset_stride = 0;
    if (offset.ndim() > 0) {
      offset_stride = static_cast<uint32_t>(offset.strides()[0]) *
          offset_storage.stride_scale;
    }

    RoPEGeneralParams params{};
    params.data[0] = dims_half;
    params.data[1] = static_cast<uint32_t>(D);
    params.data[2] = static_cast<uint32_t>(N);
    params.data[3] = static_cast<uint32_t>(B);
    params.data2[0] = scale_bits;
    params.data2[1] = base_log2_bits;
    params.data2[2] = traditional_ ? 1u : 0u;
    params.data2[3] = forward_ ? 1u : 0u;
    params.data3[0] = static_cast<uint32_t>(T);
    params.data3[1] = offset_stride;
    params.data3[2] = static_cast<uint32_t>(mat_size);
    params.data3[3] = 0;
    params.data4[0] =
        read_from_out ? 0u : shader_elem_offset(in, "[RoPE::eval_gpu] input");
    params.data4[1] = shader_elem_offset(out, "[RoPE::eval_gpu] output");
    params.data4[2] = offset_storage.offset_base;
    params.data4[3] = 0;

    auto& pool = wgpu::device().uniform_pool();
    WGPUBuffer uniform_buf = pool.acquire(
        wgpu::device().gpu_queue(), &params, sizeof(RoPEGeneralParams));

    WGPUBuffer in_buf = wgpu::wgpu_buffer(read_from_out ? out : in);
    WGPUBuffer offset_buf = wgpu::wgpu_buffer(offset);
    WGPUBuffer out_buf = wgpu::wgpu_buffer(read_from_out ? out : in);

    // When reading from output (partial/non-contiguous copy case),
    // input and output are the same buffer — but WGSL requires
    // different binding modes (read vs read_write).
    // In that case we bind the same buffer twice.
    // When reading from input, we use separate in and out buffers.
    if (read_from_out) {
      // input = output buffer (read from out, write to out)
      // We need a temporary copy for the read since WebGPU won't allow
      // the same buffer as both read-only and read-write.
      // Actually, for the partial case, we can just use the output buffer
      // for both since the rotation only modifies the first dims_ elements
      // and reads from them too. But WebGPU binding rules prevent this.
      //
      // Solution: create a temporary copy of just the data we need.
      array temp = contiguous_copy_gpu(out, s);
      encoder.add_temporary(temp);

      WGPUBuffer temp_buf = wgpu::wgpu_buffer(temp);
      uint64_t temp_buf_size = wgpu::wgpu_bind_size(temp);
      out_buf = wgpu::wgpu_buffer(out);
      uint64_t out_buf_size2 = wgpu::wgpu_bind_size(out);
      uint64_t offset_buf_size = wgpu::wgpu_bind_size(offset);

      WGPUBindGroup bg = wgpu::create_bind_group(
          pe.layout,
          {{temp_buf, temp_buf_size},
           {out_buf, out_buf_size2},
           {offset_buf, offset_buf_size},
           {uniform_buf, sizeof(RoPEGeneralParams)}});

      uint32_t total_threads =
          dims_half * static_cast<uint32_t>(T) * static_cast<uint32_t>(B * N);
      uint32_t n_workgroups =
          (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
      encoder.set_input_array(temp);
      encoder.set_input_array(offset);
      encoder.set_output_array(out);
      auto [wg_x, wg_y] =
          wgpu::get_2d_grid(n_workgroups, "[WebGPU rope_general]");
      encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);

      wgpuBindGroupRelease(bg);
    } else {
      // input != output, straightforward binding
      WGPUBuffer in_buf2 = wgpu::wgpu_buffer(in);
      uint64_t in_buf_size = wgpu::wgpu_bind_size(in);
      out_buf = wgpu::wgpu_buffer(out);
      uint64_t out_buf_size2 = wgpu::wgpu_bind_size(out);
      uint64_t offset_buf_size = wgpu::wgpu_bind_size(offset);

      WGPUBindGroup bg = wgpu::create_bind_group(
          pe.layout,
          {{in_buf2, in_buf_size},
           {out_buf, out_buf_size2},
           {offset_buf, offset_buf_size},
           {uniform_buf, sizeof(RoPEGeneralParams)}});

      uint32_t total_threads =
          dims_half * static_cast<uint32_t>(T) * static_cast<uint32_t>(B * N);
      uint32_t n_workgroups =
          (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
      encoder.set_input_array(in);
      encoder.set_input_array(offset);
      encoder.set_output_array(out);
      auto [wg_x, wg_y] =
          wgpu::get_2d_grid(n_workgroups, "[WebGPU rope_general]");
      encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);

      wgpuBindGroupRelease(bg);
    }

    encoder.add_completed_handler([uniform_buf]() {
      wgpu::device().uniform_pool().release(uniform_buf);
    });
  }
}

} // namespace mlx::core::fast
