// Copyright 2026 Apple Inc.
//
// WebGPU quantized matrix multiplication.
//
// Implements QuantizedMatmul for int4/int2/int8 packed weights with
// on-the-fly dequantization during matmul. This is critical for running
// quantized LLMs.
//
// The kernel dequantizes weights packed as u32 values:
//   value = (extracted_int - bias) * scale
// where `bias` comes from the biases array (affine quantization).
//
// QQMatmul (quantized-quantized matmul) throws not-implemented as it's
// rarely used.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

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

// Generate the quantized matmul WGSL kernel.
//
// For the non-transposed case (transpose_=false):
//   x: [B, M, K] activation
//   w: [N, K/elems_per_u32] packed weights (each u32 has elems_per_u32 int values)
//   scales: [N, K/group_size] per-group scales
//   biases: [N, K/group_size] per-group biases (zero points)
//   out: [B, M, N]
//
// For the transposed case (transpose_=true):
//   w is stored as [K/elems_per_u32, N] and we iterate K along rows
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

  int elems_per_u32 = 32 / bits;
  uint32_t mask = (1u << bits) - 1u;

  s << "const ELEMS_PER_U32: u32 = " << elems_per_u32 << "u;\n"
    << "const BIT_MASK: u32 = " << mask << "u;\n"
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

  // Extract int value at position i from packed u32
  s << "fn extract_val(packed: u32, i: u32) -> f32 {\n"
    << "  let shift = i * BITS;\n"
    << "  let val = (packed >> shift) & BIT_MASK;\n"
    << "  return f32(val);\n"
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
    << "    // Determine position in packed u32\n"
    << "    let packed_idx = k / ELEMS_PER_U32;  // which u32\n"
    << "    let elem_idx = k % ELEMS_PER_U32;    // position within u32\n"
    << "\n"
    << "    // Weight is always stored as [N, K/elems_per_u32] in row-major order.\n"
    << "    // The transpose flag affects the logical interpretation (which dim is\n"
    << "    // contracted) but NOT the physical memory layout.\n"
    << "    let w_packed = w[w_base + n * params.w_cols + packed_idx];\n"
    << "    let num_groups = params.K / params.group_size;\n"
    << "    let scale_val = f32(scales[s_base + n * num_groups + group_idx]);\n"
    << "    var bias_val: f32 = 0.0;\n"
    << "    if (params.has_bias == 1u) {\n"
    << "      bias_val = f32(biases[b_base + n * num_groups + group_idx]);\n"
    << "    }\n"
    << "\n"
    << "    // Dequantize: value = (int_val - bias) * scale\n"
    << "    let int_val = extract_val(w_packed, elem_idx);\n"
    << "    let w_dequant = (int_val - bias_val) * scale_val;\n"
    << "\n"
    << "    acc = acc + x_val * w_dequant;\n"
    << "  }\n"
    << "\n"
    << "  out[o_base + m * params.N + n] = " << x_type << "(acc);\n"
    << "}\n";

  return s.str();
}

} // namespace

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  // Only support Affine quantization mode for now
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[WebGPU QuantizedMatmul] Only Affine quantization mode is supported. "
        "Got: " +
        quantization_mode_to_string(mode_));
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

  int elems_per_u32 = 32 / bits_;
  uint32_t w_cols = K / static_cast<uint32_t>(elems_per_u32);

  const char* x_type = wgpu::dtype_to_wgsl(x.dtype());

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

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[WebGPU] QQMatmul is not implemented. "
      "This operation is rarely used.");
}

} // namespace mlx::core
