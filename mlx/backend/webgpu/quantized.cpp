// Copyright 2026 Apple Inc.
//
// WebGPU quantized matrix multiplication.
//
// Implements QuantizedMatmul for int2/int3/int4/int5/int6/int8 packed weights
// with on-the-fly dequantization during matmul. This is critical for running
// quantized LLMs.
//
// The kernel dequantizes weights packed as u32 values:
//   value = extracted_int * scale + bias
// where `bias` comes from the biases array (affine quantization).
//
// QQMatmul (quantized-quantized matmul) throws not-implemented as it's
// rarely used.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

struct QuantizedMatmulParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t bits;
  uint32_t group_size;
  uint32_t w_cols;    // number of u32 columns in packed weights
  uint32_t has_bias;  // 1 if biases present, 0 otherwise
  uint32_t batch_size;
  uint32_t batch_stride_x;
  uint32_t batch_stride_w;
  uint32_t batch_stride_scales;
  uint32_t batch_stride_biases;
  uint32_t batch_stride_out;
  uint32_t transpose;
  uint32_t _pad0;
  uint32_t _pad1;
};

struct GatherQMMParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t index_size;
  uint32_t group_size;
  uint32_t w_cols;
  uint32_t num_groups;
  uint32_t _pad0;
};

struct DequantizeParams {
  uint32_t rows;
  uint32_t K;
  uint32_t w_cols;
  uint32_t num_groups;
  uint32_t group_size;
  uint32_t out_size;
  uint32_t grid_x;
  uint32_t _pad0;
};

void validate_webgpu_quant_bits(int bits, const char* op_name) {
  if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 &&
      bits != 8) {
    throw std::runtime_error(
        std::string(op_name) +
        " supports 2, 3, 4, 5, 6, and 8-bit affine quantization on WebGPU. Got: " +
        std::to_string(bits));
  }
}

// Generate the quantized matmul WGSL kernel.
//
// Supported layout (transpose_=true):
//   x: [B, M, K] activation
//   w: [N, packed_K_u32] packed weights
//   scales: [N, K/group_size] per-group scales
//   biases: [N, K/group_size] per-group biases (zero points)
//   out: [B, M, N]
//
// Each thread computes one output element out[batch][m][n] by looping over K.
std::string make_quantized_matmul_kernel(
    const std::string& entry_name,
    const std::string& x_type,
    int bits) {
  std::ostringstream s;

  if (x_type == "f16") {
    s << "enable f16;\n\n";
  }

  uint32_t mask = (1u << bits) - 1u;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n\n";

  s << "struct QParams {\n"
    << "  M: u32, N: u32, K: u32, bits: u32,\n"
    << "  group_size: u32, w_cols: u32, has_bias: u32, batch_size: u32,\n"
    << "  batch_stride_x: u32, batch_stride_w: u32,\n"
    << "  batch_stride_scales: u32, batch_stride_biases: u32,\n"
    << "  batch_stride_out: u32, transpose: u32,\n"
    << "  _pad0: u32, _pad1: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> x: array<" << x_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read> scales: array<" << x_type << ">;\n"
    << "@group(0) @binding(3) var<storage, read> biases: array<" << x_type << ">;\n"
    << "@group(0) @binding(4) var<storage, read_write> out: array<" << x_type << ">;\n"
    << "@group(0) @binding(5) var<uniform> params: QParams;\n\n";

  // Extract int value at row-local logical K position. This treats each row
  // as one contiguous little-endian bitstream, which matches MLX's packing
  // for both power-of-two and 3/5/6-bit affine weights.
  s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
    << "  let bit_offset = k * BITS;\n"
    << "  let word_idx = bit_offset / 32u;\n"
    << "  let bit_idx = bit_offset % 32u;\n"
    << "  var val = w[row_base + word_idx] >> bit_idx;\n"
    << "  if (bit_idx + BITS > 32u) {\n"
    << "    let upper_bits = bit_idx + BITS - 32u;\n"
    << "    let upper_mask = (1u << upper_bits) - 1u;\n"
    << "    let upper = w[row_base + word_idx + 1u] & upper_mask;\n"
    << "    val = val | (upper << (BITS - upper_bits));\n"
    << "  }\n"
    << "  return f32(val & BIT_MASK);\n"
    << "}\n\n";

  // Main kernel: each thread computes one output element
  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let output_idx = gid.x;\n"
    << "  let batch = gid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let output_size = params.M * params.N;\n"
    << "  if (output_idx >= output_size) { return; }\n"
    << "\n"
    << "  let m = output_idx / params.N;\n"
    << "  let n = output_idx % params.N;\n"
    << "\n"
    << "  let x_base = batch * params.batch_stride_x;\n"
    << "  let w_base = batch * params.batch_stride_w;\n"
    << "  let s_base = batch * params.batch_stride_scales;\n"
    << "  let b_base = batch * params.batch_stride_biases;\n"
    << "  let o_base = batch * params.batch_stride_out;\n"
    << "\n"
    << "  var acc: f32 = 0.0;\n"
    << "\n"
    << "  // Loop over K dimension\n"
    << "  for (var k: u32 = 0u; k < params.K; k = k + 1u) {\n"
    << "    let x_val = f32(x[x_base + m * params.K + k]);\n"
    << "\n"
    << "    // Compute group index for this k\n"
    << "    let group_idx = k / params.group_size;\n"
    << "\n"
    << "    // Weight is always stored as [N, packed_K_u32] in row-major order.\n"
    << "    // The transpose flag affects the logical interpretation (which dim is\n"
    << "    // contracted) but NOT the physical memory layout.\n"
    << "    let w_row_base = w_base + n * params.w_cols;\n"
    << "    let num_groups = params.K / params.group_size;\n"
    << "    let scale_val = f32(scales[s_base + n * num_groups + group_idx]);\n"
    << "    var bias_val: f32 = 0.0;\n"
    << "    if (params.has_bias == 1u) {\n"
    << "      bias_val = f32(biases[b_base + n * num_groups + group_idx]);\n"
    << "    }\n"
    << "\n"
    << "    // Dequantize: value = int_val * scale + bias\n"
    << "    let int_val = extract_val(w_row_base, k);\n"
    << "    let w_dequant = int_val * scale_val + bias_val;\n"
    << "\n"
    << "    acc = acc + x_val * w_dequant;\n"
    << "  }\n"
    << "\n"
    << "  out[o_base + m * params.N + n] = " << x_type << "(acc);\n"
    << "}\n";

  return s.str();
}

std::string make_gather_qmm_kernel(
    const std::string& entry_name,
    const std::string& x_type,
    int bits) {
  std::ostringstream s;

  if (x_type == "f16") {
    s << "enable f16;\n\n";
  }

  const uint32_t mask = (1u << bits) - 1u;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n\n";

  s << "struct GatherParams {\n"
    << "  M: u32, N: u32, K: u32, index_size: u32,\n"
    << "  group_size: u32, w_cols: u32, num_groups: u32,\n"
    << "  _pad0: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> x: array<" << x_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read> scales: array<" << x_type << ">;\n"
    << "@group(0) @binding(3) var<storage, read> biases: array<" << x_type << ">;\n"
    << "@group(0) @binding(4) var<storage, read> lhs_indices: array<u32>;\n"
    << "@group(0) @binding(5) var<storage, read> rhs_indices: array<u32>;\n"
    << "@group(0) @binding(6) var<storage, read_write> out: array<" << x_type << ">;\n"
    << "@group(0) @binding(7) var<uniform> params: GatherParams;\n\n";

  s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
    << "  let bit_offset = k * BITS;\n"
    << "  let word_idx = bit_offset / 32u;\n"
    << "  let bit_idx = bit_offset % 32u;\n"
    << "  var val = w[row_base + word_idx] >> bit_idx;\n"
    << "  if (bit_idx + BITS > 32u) {\n"
    << "    let upper_bits = bit_idx + BITS - 32u;\n"
    << "    let upper_mask = (1u << upper_bits) - 1u;\n"
    << "    let upper = w[row_base + word_idx + 1u] & upper_mask;\n"
    << "    val = val | (upper << (BITS - upper_bits));\n"
    << "  }\n"
    << "  return f32(val & BIT_MASK);\n"
    << "}\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let output_idx = gid.x;\n"
    << "  let output_size = params.index_size * params.M * params.N;\n"
    << "  if (output_idx >= output_size) { return; }\n"
    << "\n"
    << "  let n = output_idx % params.N;\n"
    << "  let m = (output_idx / params.N) % params.M;\n"
    << "  let index_slot = output_idx / (params.M * params.N);\n"
    << "  let lhs_batch = lhs_indices[index_slot];\n"
    << "  let rhs_batch = rhs_indices[index_slot];\n"
    << "\n"
    << "  let x_base = (lhs_batch * params.M + m) * params.K;\n"
    << "  let w_base = (rhs_batch * params.N + n) * params.w_cols;\n"
    << "  let sb_base = (rhs_batch * params.N + n) * params.num_groups;\n"
    << "\n"
    << "  var acc: f32 = 0.0;\n"
    << "  for (var k: u32 = 0u; k < params.K; k = k + 1u) {\n"
    << "    let group_idx = k / params.group_size;\n"
    << "    let int_val = extract_val(w_base, k);\n"
    << "    let scale_val = f32(scales[sb_base + group_idx]);\n"
    << "    let bias_val = f32(biases[sb_base + group_idx]);\n"
    << "    let w_dequant = int_val * scale_val + bias_val;\n"
    << "    acc = acc + f32(x[x_base + k]) * w_dequant;\n"
    << "  }\n"
    << "\n"
    << "  out[output_idx] = " << x_type << "(acc);\n"
    << "}\n";

  return s.str();
}

std::string make_affine_dequantize_kernel(
    const std::string& entry_name,
    const std::string& out_type,
    int bits) {
  std::ostringstream s;

  if (out_type == "f16") {
    s << "enable f16;\n\n";
  }

  const uint32_t mask = (1u << bits) - 1u;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n\n";

  s << "struct DequantParams {\n"
    << "  rows: u32, K: u32, w_cols: u32, num_groups: u32,\n"
    << "  group_size: u32, out_size: u32, grid_x: u32,\n"
    << "  _pad0: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(1) var<storage, read> scales: array<" << out_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read> biases: array<" << out_type << ">;\n"
    << "@group(0) @binding(3) var<storage, read_write> out: array<" << out_type << ">;\n"
    << "@group(0) @binding(4) var<uniform> params: DequantParams;\n\n";

  s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
    << "  let bit_offset = k * BITS;\n"
    << "  let word_idx = bit_offset / 32u;\n"
    << "  let bit_idx = bit_offset % 32u;\n"
    << "  var val = w[row_base + word_idx] >> bit_idx;\n"
    << "  if (bit_idx + BITS > 32u) {\n"
    << "    let upper_bits = bit_idx + BITS - 32u;\n"
    << "    let upper_mask = (1u << upper_bits) - 1u;\n"
    << "    let upper = w[row_base + word_idx + 1u] & upper_mask;\n"
    << "    val = val | (upper << (BITS - upper_bits));\n"
    << "  }\n"
    << "  return f32(val & BIT_MASK);\n"
    << "}\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x + gid.y * params.grid_x;\n"
    << "  if (idx >= params.out_size) { return; }\n"
    << "\n"
    << "  let row = idx / params.K;\n"
    << "  let k = idx % params.K;\n"
    << "  let group_idx = k / params.group_size;\n"
    << "\n"
    << "  let int_val = extract_val(row * params.w_cols, k);\n"
    << "  let sb_idx = row * params.num_groups + group_idx;\n"
    << "  let value = int_val * f32(scales[sb_idx]) + f32(biases[sb_idx]);\n"
    << "  out[idx] = " << out_type << "(value);\n"
    << "}\n";

  return s.str();
}

} // namespace

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  validate_webgpu_quant_bits(bits_, "[WebGPU QuantizedMatmul]");

  // Only support Affine quantization mode for now
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[WebGPU QuantizedMatmul] Only Affine quantization mode is supported. "
        "Got: " +
        quantization_mode_to_string(mode_));
  }
  if (!transpose_) {
    throw std::runtime_error(
        "[WebGPU QuantizedMatmul] Only transpose=true is currently supported.");
  }

  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& biases = inputs[3];

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  if (out.size() == 0) {
    return;
  }

  uint32_t M = static_cast<uint32_t>(out.shape(-2));
  uint32_t N = static_cast<uint32_t>(out.shape(-1));
  uint32_t K = static_cast<uint32_t>(x.shape(-1));
  uint32_t batch_size = static_cast<uint32_t>(out.size() / (M * N));

  uint32_t w_cols = static_cast<uint32_t>(w.shape(-1));

  const char* x_type = wgpu::dtype_to_wgsl_safe(x.dtype());

  std::string entry_name = std::string("qmatmul_") + x_type +
      "_b" + std::to_string(bits_) +
      (transpose_ ? "_t" : "_n");
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_quantized_matmul_kernel(entry_name, x_type, bits_);
      });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(biases);
  encoder.set_output_array(out);

  // Fill params
  QuantizedMatmulParams params{};
  params.M = M;
  params.N = N;
  params.K = K;
  params.bits = static_cast<uint32_t>(bits_);
  params.group_size = static_cast<uint32_t>(group_size_);
  params.w_cols = w_cols;
  params.has_bias = 1;
  params.batch_size = batch_size;

  // Compute batch strides
  if (batch_size > 1) {
    params.batch_stride_x = M * K;
    params.batch_stride_w =
        static_cast<uint32_t>(w.size() / batch_size);
    params.batch_stride_scales =
        static_cast<uint32_t>(scales.size() / batch_size);
    params.batch_stride_biases =
        static_cast<uint32_t>(biases.size() / batch_size);
  } else {
    params.batch_stride_x = 0;
    params.batch_stride_w = 0;
    params.batch_stride_scales = 0;
    params.batch_stride_biases = 0;
  }
  params.batch_stride_out = M * N;
  params.transpose = transpose_ ? 1 : 0;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(QuantizedMatmulParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{x_buf, wgpuBufferGetSize(x_buf)},
       {w_buf, wgpuBufferGetSize(w_buf)},
       {s_buf, wgpuBufferGetSize(s_buf)},
       {b_buf, wgpuBufferGetSize(b_buf)},
       {out_buf, wgpuBufferGetSize(out_buf)},
       {uniform_buf, sizeof(QuantizedMatmulParams)}});

  uint32_t output_size = M * N;
  uint32_t num_workgroups_x =
      (output_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;

  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups_x, 1, batch_size);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  validate_webgpu_quant_bits(bits_, "[WebGPU GatherQMM]");

  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] Only Affine quantization mode is supported. Got: " +
        quantization_mode_to_string(mode_));
  }
  if (!transpose_) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] Only transpose=true is currently supported.");
  }
  if (inputs.size() != 6) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] Affine gather_qmm expects x, w, scales, biases, lhs_indices, rhs_indices.");
  }

  auto& encoder = wgpu::get_command_encoder(s);
  auto ensure_row_contiguous_matrix = [&](const array& arr) {
    if (arr.ndim() < 2) {
      return arr;
    }
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (stride_0 == arr.shape(-1) && stride_1 == 1) {
      return arr;
    }
    auto arr_copy = contiguous_copy_gpu(arr, s);
    encoder.add_temporary(arr_copy);
    return arr_copy;
  };
  auto ensure_contiguous = [&](const array& arr) {
    if (arr.flags().contiguous) {
      return arr;
    }
    auto arr_copy = contiguous_copy_gpu(arr, s);
    encoder.add_temporary(arr_copy);
    return arr_copy;
  };

  array x = ensure_row_contiguous_matrix(inputs[0]);
  array w = ensure_row_contiguous_matrix(inputs[1]);
  array scales = ensure_row_contiguous_matrix(inputs[2]);
  array biases = ensure_row_contiguous_matrix(inputs[3]);
  array lhs_indices = ensure_contiguous(inputs[4]);
  array rhs_indices = ensure_contiguous(inputs[5]);

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  if (out.size() == 0) {
    return;
  }

  uint32_t M = static_cast<uint32_t>(out.shape(-2));
  uint32_t N = static_cast<uint32_t>(out.shape(-1));
  uint32_t K = static_cast<uint32_t>(x.shape(-1));
  uint32_t index_size = static_cast<uint32_t>(out.size() / (M * N));
  uint32_t w_cols = static_cast<uint32_t>(w.shape(-1));
  uint32_t num_groups = K / static_cast<uint32_t>(group_size_);

  const char* x_type = wgpu::dtype_to_wgsl_safe(x.dtype());
  std::string entry_name = std::string("gather_qmm_") + x_type +
      "_b" + std::to_string(bits_);
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_gather_qmm_kernel(entry_name, x_type, bits_); });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(biases);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  GatherQMMParams params{};
  params.M = M;
  params.N = N;
  params.K = K;
  params.index_size = index_size;
  params.group_size = static_cast<uint32_t>(group_size_);
  params.w_cols = w_cols;
  params.num_groups = num_groups;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(GatherQMMParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases);
  WGPUBuffer lhs_buf = wgpu::wgpu_buffer(lhs_indices);
  WGPUBuffer rhs_buf = wgpu::wgpu_buffer(rhs_indices);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{x_buf, wgpuBufferGetSize(x_buf)},
       {w_buf, wgpuBufferGetSize(w_buf)},
       {s_buf, wgpuBufferGetSize(s_buf)},
       {b_buf, wgpuBufferGetSize(b_buf)},
       {lhs_buf, wgpuBufferGetSize(lhs_buf)},
       {rhs_buf, wgpuBufferGetSize(rhs_buf)},
       {out_buf, wgpuBufferGetSize(out_buf)},
       {uniform_buf, sizeof(GatherQMMParams)}});

  uint32_t output_size = static_cast<uint32_t>(out.size());
  uint32_t num_workgroups =
      (output_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  validate_webgpu_quant_bits(bits_, "[WebGPU Quantize]");

  if (!dequantize_) {
    throw std::runtime_error(
        "[WebGPU Quantize] Quantization is not implemented; affine dequantize only.");
  }
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[WebGPU Quantize] Only Affine dequantization is supported. Got: " +
        quantization_mode_to_string(mode_));
  }
  if (inputs.size() != 3 || outputs.size() != 1) {
    throw std::runtime_error(
        "[WebGPU Quantize] Affine dequantize expects w, scales, biases and one output.");
  }

  auto& encoder = wgpu::get_command_encoder(s);
  auto ensure_contiguous = [&](const array& arr) {
    if (arr.flags().contiguous) {
      return arr;
    }
    auto arr_copy = contiguous_copy_gpu(arr, s);
    encoder.add_temporary(arr_copy);
    return arr_copy;
  };

  array w = ensure_contiguous(inputs[0]);
  array scales = ensure_contiguous(inputs[1]);
  array biases = ensure_contiguous(inputs[2]);
  auto& out = outputs[0];

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  if (out.size() == 0) {
    return;
  }

  uint32_t K = static_cast<uint32_t>(out.shape(-1));
  uint32_t w_cols = static_cast<uint32_t>(w.shape(-1));
  uint32_t num_groups = K / static_cast<uint32_t>(group_size_);
  uint32_t rows = static_cast<uint32_t>(out.size() / K);

  const char* out_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  std::string entry_name = std::string("affine_dequantize_") + out_type +
      "_b" + std::to_string(bits_);
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_affine_dequantize_kernel(entry_name, out_type, bits_);
      });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(biases);
  encoder.set_output_array(out);

  DequantizeParams params{};
  params.rows = rows;
  params.K = K;
  params.w_cols = w_cols;
  params.num_groups = num_groups;
  params.group_size = static_cast<uint32_t>(group_size_);
  params.out_size = static_cast<uint32_t>(out.size());

  uint32_t num_workgroups =
      (params.out_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  constexpr uint32_t kMaxWorkgroupsPerDimension = 65535;
  uint32_t grid_x = std::min(num_workgroups, kMaxWorkgroupsPerDimension);
  uint32_t grid_y = (num_workgroups + grid_x - 1) / grid_x;
  if (grid_y > kMaxWorkgroupsPerDimension) {
    throw std::runtime_error(
        "[WebGPU Quantize] Dequantize output is too large for a 2D WebGPU dispatch.");
  }
  params.grid_x = grid_x;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(DequantizeParams));

  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{w_buf, wgpuBufferGetSize(w_buf)},
       {s_buf, wgpuBufferGetSize(s_buf)},
       {b_buf, wgpuBufferGetSize(b_buf)},
       {out_buf, wgpuBufferGetSize(out_buf)},
       {uniform_buf, sizeof(DequantizeParams)}});

  encoder.dispatch_compute(pe.pipeline, bg, grid_x, grid_y, 1);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[WebGPU] QQMatmul is not implemented. "
      "This operation is rarely used.");
}

} // namespace mlx::core
