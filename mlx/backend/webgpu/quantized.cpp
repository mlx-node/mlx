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
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mlx::core {

namespace {

constexpr uint32_t WORKGROUP_SIZE = 256;

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

WGPUBuffer create_uniform_buffer(const void* data, size_t byte_size) {
  auto& dev = wgpu::device();
  WGPUBufferDescriptor desc = {};
  desc.size = byte_size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;

  WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &desc);
  if (!buf) {
    throw std::runtime_error(
        "[WebGPU quantized] Failed to create uniform buffer");
  }
  void* mapped = wgpuBufferGetMappedRange(buf, 0, byte_size);
  std::memcpy(mapped, data, byte_size);
  wgpuBufferUnmap(buf);
  return buf;
}

WGPUShaderModule
get_shader_module(const std::string& key, const std::string& source) {
  static std::mutex mtx;
  static std::unordered_map<std::string, WGPUShaderModule> cache;

  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  auto& dev = wgpu::device();

  WGPUShaderModuleWGSLDescriptor wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgsl_desc.code = source.c_str();

  WGPUShaderModuleDescriptor desc = {};
  desc.nextInChain = &wgsl_desc.chain;
  desc.label = key.c_str();

  WGPUShaderModule mod =
      wgpuDeviceCreateShaderModule(dev.gpu_device(), &desc);
  if (!mod) {
    throw std::runtime_error(
        "[WebGPU quantized] Failed to create shader module: " + key);
  }

  cache[key] = mod;
  return mod;
}

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
    << "    // Weight layout depends on transpose flag\n"
    << "    var w_packed: u32;\n"
    << "    var scale_val: f32;\n"
    << "    var bias_val: f32;\n"
    << "    if (params.transpose == 0u) {\n"
    << "      // Non-transposed: w is [N, K/elems_per_u32]\n"
    << "      w_packed = w[w_base + n * params.w_cols + packed_idx];\n"
    << "      let num_groups = params.K / params.group_size;\n"
    << "      scale_val = f32(scales[s_base + n * num_groups + group_idx]);\n"
    << "      if (params.has_bias == 1u) {\n"
    << "        bias_val = f32(biases[b_base + n * num_groups + group_idx]);\n"
    << "      } else {\n"
    << "        bias_val = 0.0;\n"
    << "      }\n"
    << "    } else {\n"
    << "      // Transposed: w is [K/elems_per_u32, N]\n"
    << "      w_packed = w[w_base + packed_idx * params.N + n];\n"
    << "      let num_groups = params.K / params.group_size;\n"
    << "      scale_val = f32(scales[s_base + n * num_groups + group_idx]);\n"
    << "      if (params.has_bias == 1u) {\n"
    << "        bias_val = f32(biases[b_base + n * num_groups + group_idx]);\n"
    << "      } else {\n"
    << "        bias_val = 0.0;\n"
    << "      }\n"
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

  out.set_data(allocator::malloc(out.nbytes()));

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
  std::string pipeline_key = entry_name;

  std::string source =
      make_quantized_matmul_kernel(entry_name, x_type, bits_);
  WGPUShaderModule shader = get_shader_module(pipeline_key, source);

  auto& dev = wgpu::device();
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

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

  WGPUBuffer uniform_buf =
      create_uniform_buffer(&params, sizeof(QuantizedMatmulParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  // Bind group: x(0), w(1), scales(2), biases(3), out(4), params(5)
  WGPUBindGroupLayout layout =
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

  WGPUBindGroupEntry entries[6] = {};
  entries[0].binding = 0;
  entries[0].buffer = x_buf;
  entries[0].offset = 0;
  entries[0].size = wgpuBufferGetSize(x_buf);

  entries[1].binding = 1;
  entries[1].buffer = w_buf;
  entries[1].offset = 0;
  entries[1].size = wgpuBufferGetSize(w_buf);

  entries[2].binding = 2;
  entries[2].buffer = s_buf;
  entries[2].offset = 0;
  entries[2].size = wgpuBufferGetSize(s_buf);

  entries[3].binding = 3;
  entries[3].buffer = b_buf;
  entries[3].offset = 0;
  entries[3].size = wgpuBufferGetSize(b_buf);

  entries[4].binding = 4;
  entries[4].buffer = out_buf;
  entries[4].offset = 0;
  entries[4].size = wgpuBufferGetSize(out_buf);

  entries[5].binding = 5;
  entries[5].buffer = uniform_buf;
  entries[5].offset = 0;
  entries[5].size = sizeof(QuantizedMatmulParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = layout;
  bg_desc.entryCount = 6;
  bg_desc.entries = entries;

  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev.gpu_device(), &bg_desc);
  wgpuBindGroupLayoutRelease(layout);

  if (!bg) {
    throw std::runtime_error(
        "[WebGPU quantized] Failed to create bind group");
  }

  uint32_t output_size = M * N;
  uint32_t num_workgroups_x =
      (output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

  encoder.dispatch_compute(pipeline, {bg}, num_workgroups_x, 1, batch_size);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[WebGPU] QQMatmul is not implemented. "
      "This operation is rarely used.");
}

} // namespace mlx::core
