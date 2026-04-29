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

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <optional>
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
  uint32_t w_cols; // number of u32 columns in packed weights
  uint32_t has_bias; // 1 if biases present, 0 otherwise
  uint32_t batch_size;
  uint32_t batch_stride_x;
  uint32_t batch_stride_w;
  uint32_t batch_stride_scales;
  uint32_t batch_stride_biases;
  uint32_t batch_stride_out;
  uint32_t transpose;
  uint32_t offset_x;
  uint32_t offset_w;
  uint32_t offset_scales;
  uint32_t offset_biases;
  uint32_t offset_out;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};

struct GatherQMMParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t index_size;
  uint32_t group_size;
  uint32_t w_cols;
  uint32_t num_groups;
  uint32_t index_ndim;
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t _pad0;
  uint32_t _pad1;
  std::array<uint32_t, wgpu::MAX_NDIM> index_shape;
  std::array<int32_t, wgpu::MAX_NDIM> lhs_strides;
  std::array<int32_t, wgpu::MAX_NDIM> rhs_strides;
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

uint32_t shader_elem_offset(const array& arr, const char* op_name) {
  uint64_t offset = static_cast<uint64_t>(arr.offset()) /
      static_cast<uint64_t>(arr.itemsize());
  if (offset > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string(op_name) + " array offset exceeds WebGPU u32 indexing");
  }
  return static_cast<uint32_t>(offset);
}

uint32_t checked_u32(int64_t value, const char* op_name, const char* field) {
  if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string(op_name) + " " + field + " exceeds WebGPU u32 indexing");
  }
  return static_cast<uint32_t>(value);
}

int32_t checked_i32(int64_t value, const char* op_name, const char* field) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(
        std::string(op_name) + " " + field + " exceeds WebGPU i32 indexing");
  }
  return static_cast<int32_t>(value);
}

void emit_extract_val(std::ostringstream& s, int bits) {
  if (bits == 2 || bits == 4 || bits == 8) {
    uint32_t log2_bits = bits == 2 ? 4 : (bits == 4 ? 3 : 2);
    uint32_t bit_shift = bits == 2 ? 1 : (bits == 4 ? 2 : 3);
    uint32_t lane_mask = (1u << log2_bits) - 1u;
    s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
      << "  let packed = w[row_base + (k >> " << log2_bits << "u)];\n"
      << "  let shift = (k & " << lane_mask << "u) << " << bit_shift << "u;\n"
      << "  return f32((packed >> shift) & BIT_MASK);\n"
      << "}\n\n";
    return;
  }

  s << "fn load_quant_byte(row_base: u32, byte_offset: u32) -> u32 {\n"
    << "  let packed = w[row_base + (byte_offset >> 2u)];\n"
    << "  let shift = (byte_offset & 3u) << 3u;\n"
    << "  return (packed >> shift) & 0xffu;\n"
    << "}\n\n";

  if (bits == 3) {
    s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
      << "  let pack_base = (k >> 3u) * 3u;\n"
      << "  let lane = k & 7u;\n"
      << "  let b0 = load_quant_byte(row_base, pack_base);\n"
      << "  let b1 = load_quant_byte(row_base, pack_base + 1u);\n"
      << "  let b2 = load_quant_byte(row_base, pack_base + 2u);\n"
      << "  var val: u32;\n"
      << "  if (lane == 0u) {\n"
      << "    val = b0 & 0x7u;\n"
      << "  } else if (lane == 1u) {\n"
      << "    val = (b0 >> 3u) & 0x7u;\n"
      << "  } else if (lane == 2u) {\n"
      << "    val = ((b0 >> 6u) & 0x3u) | ((b1 & 0x1u) << 2u);\n"
      << "  } else if (lane == 3u) {\n"
      << "    val = (b1 >> 1u) & 0x7u;\n"
      << "  } else if (lane == 4u) {\n"
      << "    val = (b1 >> 4u) & 0x7u;\n"
      << "  } else if (lane == 5u) {\n"
      << "    val = ((b1 >> 7u) & 0x1u) | ((b2 & 0x3u) << 1u);\n"
      << "  } else if (lane == 6u) {\n"
      << "    val = (b2 >> 2u) & 0x7u;\n"
      << "  } else {\n"
      << "    val = (b2 >> 5u) & 0x7u;\n"
      << "  }\n"
      << "  return f32(val);\n"
      << "}\n\n";
    return;
  }

  if (bits == 5) {
    s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
      << "  let pack_base = (k >> 3u) * 5u;\n"
      << "  let lane = k & 7u;\n"
      << "  let b0 = load_quant_byte(row_base, pack_base);\n"
      << "  let b1 = load_quant_byte(row_base, pack_base + 1u);\n"
      << "  let b2 = load_quant_byte(row_base, pack_base + 2u);\n"
      << "  let b3 = load_quant_byte(row_base, pack_base + 3u);\n"
      << "  let b4 = load_quant_byte(row_base, pack_base + 4u);\n"
      << "  var val: u32;\n"
      << "  if (lane == 0u) {\n"
      << "    val = b0 & 0x1fu;\n"
      << "  } else if (lane == 1u) {\n"
      << "    val = ((b0 >> 5u) & 0x7u) | ((b1 & 0x3u) << 3u);\n"
      << "  } else if (lane == 2u) {\n"
      << "    val = (b1 >> 2u) & 0x1fu;\n"
      << "  } else if (lane == 3u) {\n"
      << "    val = ((b1 >> 7u) & 0x1u) | ((b2 & 0xfu) << 1u);\n"
      << "  } else if (lane == 4u) {\n"
      << "    val = ((b2 >> 4u) & 0xfu) | ((b3 & 0x1u) << 4u);\n"
      << "  } else if (lane == 5u) {\n"
      << "    val = (b3 >> 1u) & 0x1fu;\n"
      << "  } else if (lane == 6u) {\n"
      << "    val = ((b3 >> 6u) & 0x3u) | ((b4 & 0x7u) << 2u);\n"
      << "  } else {\n"
      << "    val = (b4 >> 3u) & 0x1fu;\n"
      << "  }\n"
      << "  return f32(val);\n"
      << "}\n\n";
    return;
  }

  s << "fn extract_val(row_base: u32, k: u32) -> f32 {\n"
    << "  let pack_base = (k >> 2u) * 3u;\n"
    << "  let lane = k & 3u;\n"
    << "  let b0 = load_quant_byte(row_base, pack_base);\n"
    << "  let b1 = load_quant_byte(row_base, pack_base + 1u);\n"
    << "  let b2 = load_quant_byte(row_base, pack_base + 2u);\n"
    << "  var val: u32;\n"
    << "  if (lane == 0u) {\n"
    << "    val = b0 & 0x3fu;\n"
    << "  } else if (lane == 1u) {\n"
    << "    val = ((b0 >> 6u) & 0x3u) | ((b1 & 0xfu) << 2u);\n"
    << "  } else if (lane == 2u) {\n"
    << "    val = ((b1 >> 4u) & 0xfu) | ((b2 & 0x3u) << 4u);\n"
    << "  } else {\n"
    << "    val = (b2 >> 2u) & 0x3fu;\n"
    << "  }\n"
    << "  return f32(val);\n"
    << "}\n\n";
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
    const std::string& scale_type,
    const std::string& bias_type,
    const std::string& out_type,
    int bits,
    int group_size) {
  std::ostringstream s;

  if (x_type == "f16" || scale_type == "f16" || bias_type == "f16" ||
      out_type == "f16") {
    s << "enable f16;\n\n";
  }

  uint32_t mask = (1u << bits) - 1u;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n"
    << "const GROUP_SIZE: u32 = " << group_size << "u;\n"
    << "const WG_SIZE: u32 = 256u;\n"
    << "const GRID_X: u32 = 65535u;\n\n";

  s << "struct QParams {\n"
    << "  M: u32, N: u32, K: u32, bits: u32,\n"
    << "  group_size: u32, w_cols: u32, has_bias: u32, batch_size: u32,\n"
    << "  batch_stride_x: u32, batch_stride_w: u32,\n"
    << "  batch_stride_scales: u32, batch_stride_biases: u32,\n"
    << "  batch_stride_out: u32, transpose: u32,\n"
    << "  offset_x: u32, offset_w: u32,\n"
    << "  offset_scales: u32, offset_biases: u32,\n"
    << "  offset_out: u32, _pad0: u32,\n"
    << "  _pad1: u32, _pad2: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> x: array<" << x_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read> scales: array<" << scale_type
    << ">;\n"
    << "@group(0) @binding(3) var<storage, read> biases: array<" << bias_type
    << ">;\n"
    << "@group(0) @binding(4) var<storage, read_write> out: array<" << out_type
    << ">;\n"
    << "@group(0) @binding(5) var<uniform> params: QParams;\n\n";

  emit_extract_val(s, bits);

  // Main kernel: each thread computes one output element
  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let output_idx = gid.x + gid.y * GRID_X * WG_SIZE;\n"
    << "  let batch = gid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let output_size = params.M * params.N;\n"
    << "  if (output_idx >= output_size) { return; }\n"
    << "\n"
    << "  let m = output_idx / params.N;\n"
    << "  let n = output_idx % params.N;\n"
    << "\n"
    << "  let x_base = params.offset_x + batch * params.batch_stride_x;\n"
    << "  let w_base = params.offset_w + batch * params.batch_stride_w;\n"
    << "  let s_base = params.offset_scales + batch * params.batch_stride_scales;\n"
    << "  let b_base = params.offset_biases + batch * params.batch_stride_biases;\n"
    << "  let o_base = params.offset_out + batch * params.batch_stride_out;\n"
    << "\n"
    << "  var acc: f32 = 0.0;\n"
    << "\n"
    << "  // Loop over K dimension\n"
    << "  for (var k: u32 = 0u; k < params.K; k = k + 1u) {\n"
    << "    let x_val = f32(x[x_base + m * params.K + k]);\n"
    << "\n"
    << "    // Compute group index for this k\n"
    << "    let group_idx = k / GROUP_SIZE;\n"
    << "\n"
    << "    // Weight is always stored as [N, packed_K_u32] in row-major order.\n"
    << "    // The transpose flag affects the logical interpretation (which dim is\n"
    << "    // contracted) but NOT the physical memory layout.\n"
    << "    let w_row_base = w_base + n * params.w_cols;\n"
    << "    let num_groups = params.K / GROUP_SIZE;\n"
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
    << "  out[o_base + m * params.N + n] = " << out_type << "(acc);\n"
    << "}\n";

  return s.str();
}

// Decode GEMV hot path: each workgroup computes eight adjacent output columns
// for one row. Threads cooperate over K, which avoids the scalar "one thread
// loops over all K" bottleneck in the generic qmatmul kernel.
void emit_gemv_multi_col_reduction(
    std::ostringstream& s,
    const std::string& out_type,
    int subgroup_size,
    int cols_per_wg = 8);

std::string make_quantized_gemv_kernel(
    const std::string& entry_name,
    const std::string& x_type,
    const std::string& scale_type,
    const std::string& bias_type,
    const std::string& out_type,
    int bits,
    int group_size,
    int subgroup_size) {
  std::ostringstream s;

  if (subgroup_size > 0) {
    s << "enable subgroups;\n";
  }
  if (x_type == "f16" || scale_type == "f16" || bias_type == "f16" ||
      out_type == "f16") {
    s << "enable f16;\n\n";
  } else if (subgroup_size > 0) {
    s << "\n";
  }

  uint32_t mask = (1u << bits) - 1u;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n"
    << "const GROUP_SIZE: u32 = " << group_size << "u;\n"
    << "const WG_SIZE: u32 = 128u;\n"
    << "const COLS_PER_WG: u32 = 8u;\n"
    << "const GRID_X: u32 = 65535u;\n\n";

  s << "struct QParams {\n"
    << "  M: u32, N: u32, K: u32, bits: u32,\n"
    << "  group_size: u32, w_cols: u32, has_bias: u32, batch_size: u32,\n"
    << "  batch_stride_x: u32, batch_stride_w: u32,\n"
    << "  batch_stride_scales: u32, batch_stride_biases: u32,\n"
    << "  batch_stride_out: u32, transpose: u32,\n"
    << "  offset_x: u32, offset_w: u32,\n"
    << "  offset_scales: u32, offset_biases: u32,\n"
    << "  offset_out: u32, _pad0: u32,\n"
    << "  _pad1: u32, _pad2: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> x: array<" << x_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read> scales: array<" << scale_type
    << ">;\n"
    << "@group(0) @binding(3) var<storage, read> biases: array<" << bias_type
    << ">;\n"
    << "@group(0) @binding(4) var<storage, read_write> out: array<" << out_type
    << ">;\n"
    << "@group(0) @binding(5) var<uniform> params: QParams;\n\n";

  s << "fn sum_op(x: f32, y: f32) -> f32 { return x + y; }\n\n";
  s << "var<workgroup> shared0: array<f32, 128>;\n"
    << "var<workgroup> shared1: array<f32, 128>;\n"
    << "var<workgroup> shared2: array<f32, 128>;\n"
    << "var<workgroup> shared3: array<f32, 128>;\n"
    << "var<workgroup> shared4: array<f32, 128>;\n"
    << "var<workgroup> shared5: array<f32, 128>;\n"
    << "var<workgroup> shared6: array<f32, 128>;\n"
    << "var<workgroup> shared7: array<f32, 128>;\n\n";

  emit_extract_val(s, bits);

  s << "@compute @workgroup_size(128)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wid: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u) {\n"
    << "  let tile_idx = wid.x + wid.y * GRID_X;\n"
    << "  let batch = wid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let num_col_tiles = (params.N + COLS_PER_WG - 1u) / COLS_PER_WG;\n"
    << "  let total_tiles = params.M * num_col_tiles;\n"
    << "  if (tile_idx >= total_tiles) { return; }\n"
    << "  let m = tile_idx / num_col_tiles;\n"
    << "  let col_base = (tile_idx % num_col_tiles) * COLS_PER_WG;\n"
    << "  let col0 = col_base;\n"
    << "  let col1 = col_base + 1u;\n"
    << "  let col2 = col_base + 2u;\n"
    << "  let col3 = col_base + 3u;\n"
    << "  let col4 = col_base + 4u;\n"
    << "  let col5 = col_base + 5u;\n"
    << "  let col6 = col_base + 6u;\n"
    << "  let col7 = col_base + 7u;\n"
    << "  let has1 = col1 < params.N;\n"
    << "  let has2 = col2 < params.N;\n"
    << "  let has3 = col3 < params.N;\n"
    << "  let has4 = col4 < params.N;\n"
    << "  let has5 = col5 < params.N;\n"
    << "  let has6 = col6 < params.N;\n"
    << "  let has7 = col7 < params.N;\n"
    << "  let tid = lid.x;\n"
    << "  let x_base = params.offset_x + batch * params.batch_stride_x;\n"
    << "  let w_base = params.offset_w + batch * params.batch_stride_w;\n"
    << "  let s_base = params.offset_scales + batch * params.batch_stride_scales;\n"
    << "  let b_base = params.offset_biases + batch * params.batch_stride_biases;\n"
    << "  let o_base = params.offset_out + batch * params.batch_stride_out;\n"
    << "  let num_groups = params.K / GROUP_SIZE;\n"
    << "  var acc0: f32 = 0.0;\n"
    << "  var acc1: f32 = 0.0;\n"
    << "  var acc2: f32 = 0.0;\n"
    << "  var acc3: f32 = 0.0;\n"
    << "  var acc4: f32 = 0.0;\n"
    << "  var acc5: f32 = 0.0;\n"
    << "  var acc6: f32 = 0.0;\n"
    << "  var acc7: f32 = 0.0;\n"
    << "\n"
    << "  for (var k: u32 = tid; k < params.K; k = k + WG_SIZE) {\n"
    << "    let x_val = f32(x[x_base + m * params.K + k]);\n"
    << "    let group_idx = k / GROUP_SIZE;\n"
    << "\n"
    << "    let row0 = w_base + col0 * params.w_cols;\n"
    << "    let sb0 = s_base + col0 * num_groups + group_idx;\n"
    << "    var bias0: f32 = 0.0;\n"
    << "    if (params.has_bias == 1u) { bias0 = f32(biases[b_base + col0 * num_groups + group_idx]); }\n"
    << "    let w0 = extract_val(row0, k) * f32(scales[sb0]) + bias0;\n"
    << "    acc0 = acc0 + x_val * w0;\n"
    << "\n"
    << "    if (has1) {\n"
    << "      let row1 = w_base + col1 * params.w_cols;\n"
    << "      let sb1 = s_base + col1 * num_groups + group_idx;\n"
    << "      var bias1: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias1 = f32(biases[b_base + col1 * num_groups + group_idx]); }\n"
    << "      let w1 = extract_val(row1, k) * f32(scales[sb1]) + bias1;\n"
    << "      acc1 = acc1 + x_val * w1;\n"
    << "    }\n"
    << "    if (has2) {\n"
    << "      let row2 = w_base + col2 * params.w_cols;\n"
    << "      let sb2 = s_base + col2 * num_groups + group_idx;\n"
    << "      var bias2: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias2 = f32(biases[b_base + col2 * num_groups + group_idx]); }\n"
    << "      let w2 = extract_val(row2, k) * f32(scales[sb2]) + bias2;\n"
    << "      acc2 = acc2 + x_val * w2;\n"
    << "    }\n"
    << "    if (has3) {\n"
    << "      let row3 = w_base + col3 * params.w_cols;\n"
    << "      let sb3 = s_base + col3 * num_groups + group_idx;\n"
    << "      var bias3: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias3 = f32(biases[b_base + col3 * num_groups + group_idx]); }\n"
    << "      let w3 = extract_val(row3, k) * f32(scales[sb3]) + bias3;\n"
    << "      acc3 = acc3 + x_val * w3;\n"
    << "    }\n"
    << "    if (has4) {\n"
    << "      let row4 = w_base + col4 * params.w_cols;\n"
    << "      let sb4 = s_base + col4 * num_groups + group_idx;\n"
    << "      var bias4: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias4 = f32(biases[b_base + col4 * num_groups + group_idx]); }\n"
    << "      let w4 = extract_val(row4, k) * f32(scales[sb4]) + bias4;\n"
    << "      acc4 = acc4 + x_val * w4;\n"
    << "    }\n"
    << "    if (has5) {\n"
    << "      let row5 = w_base + col5 * params.w_cols;\n"
    << "      let sb5 = s_base + col5 * num_groups + group_idx;\n"
    << "      var bias5: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias5 = f32(biases[b_base + col5 * num_groups + group_idx]); }\n"
    << "      let w5 = extract_val(row5, k) * f32(scales[sb5]) + bias5;\n"
    << "      acc5 = acc5 + x_val * w5;\n"
    << "    }\n"
    << "    if (has6) {\n"
    << "      let row6 = w_base + col6 * params.w_cols;\n"
    << "      let sb6 = s_base + col6 * num_groups + group_idx;\n"
    << "      var bias6: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias6 = f32(biases[b_base + col6 * num_groups + group_idx]); }\n"
    << "      let w6 = extract_val(row6, k) * f32(scales[sb6]) + bias6;\n"
    << "      acc6 = acc6 + x_val * w6;\n"
    << "    }\n"
    << "    if (has7) {\n"
    << "      let row7 = w_base + col7 * params.w_cols;\n"
    << "      let sb7 = s_base + col7 * num_groups + group_idx;\n"
    << "      var bias7: f32 = 0.0;\n"
    << "      if (params.has_bias == 1u) { bias7 = f32(biases[b_base + col7 * num_groups + group_idx]); }\n"
    << "      let w7 = extract_val(row7, k) * f32(scales[sb7]) + bias7;\n"
    << "      acc7 = acc7 + x_val * w7;\n"
    << "    }\n"
    << "  }\n"
    << "\n";
  emit_gemv_multi_col_reduction(s, out_type, subgroup_size);

  return s.str();
}

void emit_q4_word_accumulate(std::ostringstream& s, int col) {
  const std::string suffix = std::to_string(col);
  if (col > 0) {
    s << "    if (has" << suffix << ") {\n";
  }
  const std::string indent = col > 0 ? "      " : "    ";
  s << indent << "let row" << suffix << " = w_base + col" << suffix
    << " * params.w_cols;\n"
    << indent << "let sb" << suffix << " = s_base + col" << suffix
    << " * num_groups + group_idx;\n"
    << indent << "var bias" << suffix << ": f32 = 0.0;\n"
    << indent << "if (params.has_bias == 1u) { bias" << suffix
    << " = f32(biases[b_base + col" << suffix
    << " * num_groups + group_idx]); }\n"
    << indent << "acc" << suffix << " = acc" << suffix << " + q4_dot_word(w[row"
    << suffix << " + word_idx], f32(scales[sb" << suffix << "]), bias" << suffix
    << ", xlo, xhi);\n";
  if (col > 0) {
    s << "    }\n";
  }
}

void emit_gemv_multi_col_reduction(
    std::ostringstream& s,
    const std::string& out_type,
    int subgroup_size,
    int cols_per_wg) {
  if (subgroup_size > 0) {
    const uint32_t sg = static_cast<uint32_t>(subgroup_size);
    const uint32_t num_partials = 128u / sg;
    for (int i = 0; i < cols_per_wg; ++i) {
      s << "  var sg" << i << " = subgroupAdd(acc" << i << ");\n";
    }
    s << "  let sg_id = tid / " << sg << "u;\n"
      << "  if (subgroupElect()) {\n";
    for (int i = 0; i < cols_per_wg; ++i) {
      s << "    shared" << i << "[sg_id] = sg" << i << ";\n";
    }
    s << "  }\n"
      << "  workgroupBarrier();\n";
    for (uint32_t stride = num_partials >> 1; stride >= 1; stride >>= 1) {
      s << "  if (tid < " << stride << "u) {\n";
      for (int i = 0; i < cols_per_wg; ++i) {
        s << "    shared" << i << "[tid] = shared" << i << "[tid] + shared" << i
          << "[tid + " << stride << "u];\n";
      }
      s << "  }\n"
        << "  workgroupBarrier();\n";
    }
  } else {
    for (int i = 0; i < cols_per_wg; ++i) {
      s << "  shared" << i << "[tid] = acc" << i << ";\n";
    }
    s << "  workgroupBarrier();\n";
    for (uint32_t stride = 64; stride >= 1; stride >>= 1) {
      s << "  if (tid < " << stride << "u) {\n";
      for (int i = 0; i < cols_per_wg; ++i) {
        s << "    shared" << i << "[tid] = shared" << i << "[tid] + shared" << i
          << "[tid + " << stride << "u];\n";
      }
      s << "  }\n"
        << "  workgroupBarrier();\n";
    }
  }

  s << "  if (tid == 0u) {\n"
    << "    let out_row = o_base + m * params.N;\n"
    << "    out[out_row + col0] = " << out_type << "(shared0[0]);\n";
  for (int i = 1; i < cols_per_wg; ++i) {
    s << "    if (has" << i << ") { out[out_row + col" << i
      << "] = " << out_type << "(shared" << i << "[0]); }\n";
  }
  s << "  }\n"
    << "}\n";
}

// Q4 decode GEMV hot path. Each lane consumes one packed u32 word (8 weights)
// instead of one logical K value. That avoids eight lanes rereading the same
// packed word and reloads each affine scale/bias once per quantization group
// word rather than once per nibble.
std::string make_quantized_q4_gemv_kernel(
    const std::string& entry_name,
    const std::string& x_type,
    const std::string& scale_type,
    const std::string& bias_type,
    const std::string& out_type,
    int group_size,
    int subgroup_size) {
  std::ostringstream s;
  constexpr int kColsPerWorkgroup = 8;

  if (subgroup_size > 0) {
    s << "enable subgroups;\n";
  }
  if (x_type == "f16" || scale_type == "f16" || bias_type == "f16" ||
      out_type == "f16") {
    s << "enable f16;\n\n";
  } else if (subgroup_size > 0) {
    s << "\n";
  }

  s << "const BIT_MASK: u32 = 0xFu;\n"
    << "const GROUP_SIZE: u32 = " << group_size << "u;\n"
    << "const WG_SIZE: u32 = 128u;\n"
    << "const COLS_PER_WG: u32 = " << kColsPerWorkgroup << "u;\n"
    << "const GRID_X: u32 = 65535u;\n\n";

  s << "struct QParams {\n"
    << "  M: u32, N: u32, K: u32, bits: u32,\n"
    << "  group_size: u32, w_cols: u32, has_bias: u32, batch_size: u32,\n"
    << "  batch_stride_x: u32, batch_stride_w: u32,\n"
    << "  batch_stride_scales: u32, batch_stride_biases: u32,\n"
    << "  batch_stride_out: u32, transpose: u32,\n"
    << "  offset_x: u32, offset_w: u32,\n"
    << "  offset_scales: u32, offset_biases: u32,\n"
    << "  offset_out: u32, _pad0: u32,\n"
    << "  _pad1: u32, _pad2: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> x: array<" << x_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read> scales: array<" << scale_type
    << ">;\n"
    << "@group(0) @binding(3) var<storage, read> biases: array<" << bias_type
    << ">;\n"
    << "@group(0) @binding(4) var<storage, read_write> out: array<" << out_type
    << ">;\n"
    << "@group(0) @binding(5) var<uniform> params: QParams;\n\n";

  s << "fn sum_op(x: f32, y: f32) -> f32 { return x + y; }\n\n"
    << "fn q4_dot_word(packed: u32, scale: f32, bias: f32, "
    << "xlo: vec4<f32>, xhi: vec4<f32>) -> f32 {\n"
    << "  let lo = vec4<f32>(\n"
    << "    f32(packed & BIT_MASK),\n"
    << "    f32((packed >> 4u) & BIT_MASK),\n"
    << "    f32((packed >> 8u) & BIT_MASK),\n"
    << "    f32((packed >> 12u) & BIT_MASK)) * scale + vec4<f32>(bias);\n"
    << "  let hi = vec4<f32>(\n"
    << "    f32((packed >> 16u) & BIT_MASK),\n"
    << "    f32((packed >> 20u) & BIT_MASK),\n"
    << "    f32((packed >> 24u) & BIT_MASK),\n"
    << "    f32((packed >> 28u) & BIT_MASK)) * scale + vec4<f32>(bias);\n"
    << "  return dot(xlo, lo) + dot(xhi, hi);\n"
    << "}\n\n";

  for (int i = 0; i < kColsPerWorkgroup; ++i) {
    s << "var<workgroup> shared" << i << ": array<f32, 128>;\n";
  }
  s << "\n";

  s << "@compute @workgroup_size(128)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wid: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u) {\n"
    << "  let tile_idx = wid.x + wid.y * GRID_X;\n"
    << "  let batch = wid.z;\n"
    << "  if (batch >= params.batch_size) { return; }\n"
    << "  let num_col_tiles = (params.N + COLS_PER_WG - 1u) / COLS_PER_WG;\n"
    << "  let total_tiles = params.M * num_col_tiles;\n"
    << "  if (tile_idx >= total_tiles) { return; }\n"
    << "  let m = tile_idx / num_col_tiles;\n"
    << "  let col_base = (tile_idx % num_col_tiles) * COLS_PER_WG;\n"
    << "  let col0 = col_base;\n";
  for (int i = 1; i < kColsPerWorkgroup; ++i) {
    s << "  let col" << i << " = col_base + " << i << "u;\n";
  }
  for (int i = 1; i < kColsPerWorkgroup; ++i) {
    s << "  let has" << i << " = col" << i << " < params.N;\n";
  }
  s << "  let tid = lid.x;\n"
    << "  let x_base = params.offset_x + batch * params.batch_stride_x + m * params.K;\n"
    << "  let w_base = params.offset_w + batch * params.batch_stride_w;\n"
    << "  let s_base = params.offset_scales + batch * params.batch_stride_scales;\n"
    << "  let b_base = params.offset_biases + batch * params.batch_stride_biases;\n"
    << "  let o_base = params.offset_out + batch * params.batch_stride_out;\n"
    << "  let num_groups = params.K / GROUP_SIZE;\n"
    << "  var acc0: f32 = 0.0;\n";
  for (int i = 1; i < kColsPerWorkgroup; ++i) {
    s << "  var acc" << i << ": f32 = 0.0;\n";
  }
  s << "\n"
    << "  for (var word_idx: u32 = tid; word_idx < params.w_cols; "
    << "word_idx = word_idx + WG_SIZE) {\n"
    << "    let k_base = word_idx << 3u;\n"
    << "    let group_idx = k_base / GROUP_SIZE;\n"
    << "    let xlo = vec4<f32>(\n"
    << "      f32(x[x_base + k_base]),\n"
    << "      f32(x[x_base + k_base + 1u]),\n"
    << "      f32(x[x_base + k_base + 2u]),\n"
    << "      f32(x[x_base + k_base + 3u]));\n"
    << "    let xhi = vec4<f32>(\n"
    << "      f32(x[x_base + k_base + 4u]),\n"
    << "      f32(x[x_base + k_base + 5u]),\n"
    << "      f32(x[x_base + k_base + 6u]),\n"
    << "      f32(x[x_base + k_base + 7u]));\n";
  for (int col = 0; col < kColsPerWorkgroup; ++col) {
    emit_q4_word_accumulate(s, col);
  }
  s << "  }\n"
    << "\n";
  emit_gemv_multi_col_reduction(s, out_type, subgroup_size, kColsPerWorkgroup);

  return s.str();
}

std::string make_gather_qmm_kernel(
    const std::string& entry_name,
    const std::string& x_type,
    const std::string& scale_type,
    const std::string& bias_type,
    const std::string& out_type,
    int bits,
    int group_size) {
  std::ostringstream s;

  if (x_type == "f16" || scale_type == "f16" || bias_type == "f16" ||
      out_type == "f16") {
    s << "enable f16;\n\n";
  }

  const uint32_t mask = (1u << bits) - 1u;
  constexpr uint32_t kIndexVecs = (wgpu::MAX_NDIM + 3) / 4;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n"
    << "const GROUP_SIZE: u32 = " << group_size << "u;\n"
    << "const WG_SIZE: u32 = 256u;\n"
    << "const GRID_X: u32 = 65535u;\n\n";

  s << "struct GatherParams {\n"
    << "  M: u32, N: u32, K: u32, index_size: u32,\n"
    << "  group_size: u32, w_cols: u32, num_groups: u32,\n"
    << "  index_ndim: u32,\n"
    << "  lhs_offset: u32, rhs_offset: u32,\n"
    << "  _pad0: u32, _pad1: u32,\n"
    << "  index_shape: array<vec4u, " << kIndexVecs << ">,\n"
    << "  lhs_strides: array<vec4i, " << kIndexVecs << ">,\n"
    << "  rhs_strides: array<vec4i, " << kIndexVecs << ">,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> x: array<" << x_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read> scales: array<" << scale_type
    << ">;\n"
    << "@group(0) @binding(3) var<storage, read> biases: array<" << bias_type
    << ">;\n"
    << "@group(0) @binding(4) var<storage, read> lhs_indices: array<u32>;\n"
    << "@group(0) @binding(5) var<storage, read> rhs_indices: array<u32>;\n"
    << "@group(0) @binding(6) var<storage, read_write> out: array<" << out_type
    << ">;\n"
    << "@group(0) @binding(7) var<uniform> params: GatherParams;\n\n";

  emit_extract_val(s, bits);

  s << "fn get_index_shape(d: u32) -> u32 {\n"
    << "  let v = params.index_shape[d >> 2u];\n"
    << "  switch (d & 3u) {\n"
    << "    case 0u: { return v.x; }\n"
    << "    case 1u: { return v.y; }\n"
    << "    case 2u: { return v.z; }\n"
    << "    default: { return v.w; }\n"
    << "  }\n"
    << "}\n\n"
    << "fn get_lhs_stride(d: u32) -> i32 {\n"
    << "  let v = params.lhs_strides[d >> 2u];\n"
    << "  switch (d & 3u) {\n"
    << "    case 0u: { return v.x; }\n"
    << "    case 1u: { return v.y; }\n"
    << "    case 2u: { return v.z; }\n"
    << "    default: { return v.w; }\n"
    << "  }\n"
    << "}\n\n"
    << "fn get_rhs_stride(d: u32) -> i32 {\n"
    << "  let v = params.rhs_strides[d >> 2u];\n"
    << "  switch (d & 3u) {\n"
    << "    case 0u: { return v.x; }\n"
    << "    case 1u: { return v.y; }\n"
    << "    case 2u: { return v.z; }\n"
    << "    default: { return v.w; }\n"
    << "  }\n"
    << "}\n\n"
    << "fn lhs_index_loc(flat_idx: u32) -> u32 {\n"
    << "  var loc: i32 = i32(params.lhs_offset);\n"
    << "  var rem = flat_idx;\n"
    << "  for (var r: u32 = 0u; r < params.index_ndim; r = r + 1u) {\n"
    << "    let d = params.index_ndim - 1u - r;\n"
    << "    let dim = get_index_shape(d);\n"
    << "    let coord = rem % dim;\n"
    << "    rem = rem / dim;\n"
    << "    loc = loc + i32(coord) * get_lhs_stride(d);\n"
    << "  }\n"
    << "  return u32(loc);\n"
    << "}\n\n"
    << "fn rhs_index_loc(flat_idx: u32) -> u32 {\n"
    << "  var loc: i32 = i32(params.rhs_offset);\n"
    << "  var rem = flat_idx;\n"
    << "  for (var r: u32 = 0u; r < params.index_ndim; r = r + 1u) {\n"
    << "    let d = params.index_ndim - 1u - r;\n"
    << "    let dim = get_index_shape(d);\n"
    << "    let coord = rem % dim;\n"
    << "    rem = rem / dim;\n"
    << "    loc = loc + i32(coord) * get_rhs_stride(d);\n"
    << "  }\n"
    << "  return u32(loc);\n"
    << "}\n\n"
    << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let output_idx = gid.x + gid.y * GRID_X * WG_SIZE;\n"
    << "  let output_size = params.index_size * params.M * params.N;\n"
    << "  if (output_idx >= output_size) { return; }\n"
    << "\n"
    << "  let n = output_idx % params.N;\n"
    << "  let m = (output_idx / params.N) % params.M;\n"
    << "  let index_slot = output_idx / (params.M * params.N);\n"
    << "  let lhs_batch = lhs_indices[lhs_index_loc(index_slot)];\n"
    << "  let rhs_batch = rhs_indices[rhs_index_loc(index_slot)];\n"
    << "\n"
    << "  let x_base = (lhs_batch * params.M + m) * params.K;\n"
    << "  let w_base = (rhs_batch * params.N + n) * params.w_cols;\n"
    << "  let sb_base = (rhs_batch * params.N + n) * params.num_groups;\n"
    << "\n"
    << "  var acc: f32 = 0.0;\n"
    << "  for (var k: u32 = 0u; k < params.K; k = k + 1u) {\n"
    << "    let group_idx = k / GROUP_SIZE;\n"
    << "    let int_val = extract_val(w_base, k);\n"
    << "    let scale_val = f32(scales[sb_base + group_idx]);\n"
    << "    let bias_val = f32(biases[sb_base + group_idx]);\n"
    << "    let w_dequant = int_val * scale_val + bias_val;\n"
    << "    acc = acc + f32(x[x_base + k]) * w_dequant;\n"
    << "  }\n"
    << "\n"
    << "  out[output_idx] = " << out_type << "(acc);\n"
    << "}\n";

  return s.str();
}

std::string make_affine_dequantize_kernel(
    const std::string& entry_name,
    const std::string& out_type,
    const std::string& scale_type,
    const std::string& bias_type,
    int bits,
    int group_size) {
  std::ostringstream s;

  if (out_type == "f16" || scale_type == "f16" || bias_type == "f16") {
    s << "enable f16;\n\n";
  }

  const uint32_t mask = (1u << bits) - 1u;

  s << "const BIT_MASK: u32 = " << mask << "u;\n"
    << "const BITS: u32 = " << bits << "u;\n"
    << "const GROUP_SIZE: u32 = " << group_size << "u;\n"
    << "const WG_SIZE: u32 = 256u;\n\n";

  s << "struct DequantParams {\n"
    << "  rows: u32, K: u32, w_cols: u32, num_groups: u32,\n"
    << "  group_size: u32, out_size: u32, grid_x: u32,\n"
    << "  _pad0: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> w: array<u32>;\n"
    << "@group(0) @binding(1) var<storage, read> scales: array<" << scale_type
    << ">;\n"
    << "@group(0) @binding(2) var<storage, read> biases: array<" << bias_type
    << ">;\n"
    << "@group(0) @binding(3) var<storage, read_write> out: array<" << out_type
    << ">;\n"
    << "@group(0) @binding(4) var<uniform> params: DequantParams;\n\n";

  emit_extract_val(s, bits);

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x + gid.y * params.grid_x * WG_SIZE;\n"
    << "  if (idx >= params.out_size) { return; }\n"
    << "\n"
    << "  let row = idx / params.K;\n"
    << "  let k = idx % params.K;\n"
    << "  let group_idx = k / GROUP_SIZE;\n"
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

  auto& encoder = wgpu::get_command_encoder(s);
  auto ensure_row_contiguous_matrix = [&](const array& arr) {
    if (arr.ndim() < 2) {
      if (arr.flags().contiguous) {
        return arr;
      }
      auto arr_copy = contiguous_copy_gpu(arr, s);
      encoder.add_temporary(arr_copy);
      return arr_copy;
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

  array x_contiguous = ensure_row_contiguous_matrix(x);
  array w_contiguous = ensure_row_contiguous_matrix(w);
  array scales_contiguous = ensure_row_contiguous_matrix(scales);
  array biases_contiguous = ensure_row_contiguous_matrix(biases);

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  if (out.size() == 0) {
    return;
  }

  uint32_t M = static_cast<uint32_t>(out.shape(-2));
  uint32_t N = static_cast<uint32_t>(out.shape(-1));
  uint32_t K = static_cast<uint32_t>(x_contiguous.shape(-1));
  uint32_t batch_size = static_cast<uint32_t>(out.size() / (M * N));

  auto flat_batch_stride = [&](const array& arr,
                               const char* name) -> std::optional<uint32_t> {
    if (batch_size <= 1 || arr.ndim() < 3) {
      return 0;
    }

    int64_t logical_batch = 1;
    for (int i = 0; i < arr.ndim() - 2; ++i) {
      logical_batch *= arr.shape()[i];
    }
    if (logical_batch == 1) {
      return 0;
    }
    if (logical_batch != batch_size) {
      return std::nullopt;
    }

    int64_t expected_stride = arr.shape(-2) * arr.shape(-1);
    int64_t flat_stride = expected_stride;
    for (int i = arr.ndim() - 3; i >= 0; --i) {
      if (arr.shape()[i] != 1) {
        if (arr.strides()[i] != expected_stride) {
          return std::nullopt;
        }
        flat_stride = arr.strides()[i];
      }
      expected_stride *= arr.shape()[i];
    }

    return checked_u32(flat_stride, "[WebGPU QuantizedMatmul]", name);
  };

  auto ensure_linear_flat_batch = [&](const array& arr, const char* name) {
    if (flat_batch_stride(arr, name).has_value()) {
      return arr;
    }
    auto arr_copy = contiguous_copy_gpu(arr, s);
    encoder.add_temporary(arr_copy);
    return arr_copy;
  };

  x_contiguous = ensure_linear_flat_batch(x_contiguous, "batch_stride_x");
  w_contiguous = ensure_linear_flat_batch(w_contiguous, "batch_stride_w");
  scales_contiguous =
      ensure_linear_flat_batch(scales_contiguous, "batch_stride_scales");
  biases_contiguous =
      ensure_linear_flat_batch(biases_contiguous, "batch_stride_biases");

  uint32_t w_cols = static_cast<uint32_t>(w_contiguous.shape(-1));

  const char* x_type = wgpu::dtype_to_wgsl_safe(x_contiguous.dtype());
  const char* scale_type = wgpu::dtype_to_wgsl_safe(scales_contiguous.dtype());
  const char* bias_type = wgpu::dtype_to_wgsl_safe(biases_contiguous.dtype());
  const char* out_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  const bool use_gemv = (M == 1);
  const bool use_q4_word_gemv =
      use_gemv && bits_ == 4 && (group_size_ % 8 == 0) && (K % 8 == 0);
  int subgroup_size = use_gemv ? wgpu::effective_subgroup_size() : 0;
  if (subgroup_size > 0 && (128 % subgroup_size) != 0) {
    subgroup_size = 0;
  }

  std::string entry_name =
      std::string(
          use_q4_word_gemv ? "qgemv4w_" : (use_gemv ? "qgemv_" : "qmatmul_")) +
      x_type + "_s" + scale_type + "_z" + bias_type + "_o" + out_type + "_b" +
      std::to_string(bits_) + "_g" + std::to_string(group_size_) +
      (subgroup_size > 0 ? "_sg" + std::to_string(subgroup_size) : "") +
      (transpose_ ? "_t" : "_n");
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(entry_name, [&]() {
    if (use_q4_word_gemv) {
      return make_quantized_q4_gemv_kernel(
          entry_name,
          x_type,
          scale_type,
          bias_type,
          out_type,
          group_size_,
          subgroup_size);
    } else if (use_gemv) {
      return make_quantized_gemv_kernel(
          entry_name,
          x_type,
          scale_type,
          bias_type,
          out_type,
          bits_,
          group_size_,
          subgroup_size);
    }
    return make_quantized_matmul_kernel(
        entry_name,
        x_type,
        scale_type,
        bias_type,
        out_type,
        bits_,
        group_size_);
  });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(x_contiguous);
  encoder.set_input_array(w_contiguous);
  encoder.set_input_array(scales_contiguous);
  encoder.set_input_array(biases_contiguous);
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

  // Compute flattened batch strides from each input independently. A 2D
  // quantized weight/scales/biases tensor is broadcast across activation
  // batches and must therefore use stride 0.
  params.batch_stride_x =
      flat_batch_stride(x_contiguous, "batch_stride_x").value();
  params.batch_stride_w =
      flat_batch_stride(w_contiguous, "batch_stride_w").value();
  params.batch_stride_scales =
      flat_batch_stride(scales_contiguous, "batch_stride_scales").value();
  params.batch_stride_biases =
      flat_batch_stride(biases_contiguous, "batch_stride_biases").value();
  params.batch_stride_out = M * N;
  params.transpose = transpose_ ? 1 : 0;
  params.offset_x =
      shader_elem_offset(x_contiguous, "[WebGPU QuantizedMatmul]");
  params.offset_w =
      shader_elem_offset(w_contiguous, "[WebGPU QuantizedMatmul]");
  params.offset_scales =
      shader_elem_offset(scales_contiguous, "[WebGPU QuantizedMatmul]");
  params.offset_biases =
      shader_elem_offset(biases_contiguous, "[WebGPU QuantizedMatmul]");
  params.offset_out = shader_elem_offset(out, "[WebGPU QuantizedMatmul]");

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf = pool.acquire(
      wgpu::device().gpu_queue(), &params, sizeof(QuantizedMatmulParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x_contiguous);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w_contiguous);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales_contiguous);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases_contiguous);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{x_buf, wgpu::wgpu_bind_size(x_contiguous)},
       {w_buf, wgpu::wgpu_bind_size(w_contiguous)},
       {s_buf, wgpu::wgpu_bind_size(scales_contiguous)},
       {b_buf, wgpu::wgpu_bind_size(biases_contiguous)},
       {out_buf, wgpu::wgpu_bind_size(out)},
       {uniform_buf, sizeof(QuantizedMatmulParams)}});

  if (use_gemv) {
    constexpr uint32_t kColsPerWorkgroup = 8;
    constexpr uint32_t kGridX = 65535u;
    uint32_t col_tiles = (N + kColsPerWorkgroup - 1) / kColsPerWorkgroup;
    const uint64_t total_tiles_u64 =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(col_tiles);
    constexpr uint64_t kGridMax2D =
        static_cast<uint64_t>(kGridX) * static_cast<uint64_t>(kGridX);
    if (total_tiles_u64 >= kGridMax2D) {
      throw std::runtime_error(
          "[WebGPU QuantizedMatmul] GEMV output is too large for a 2D WebGPU dispatch.");
    }
    uint32_t total_tiles = static_cast<uint32_t>(total_tiles_u64);
    uint32_t wg_x = std::min(total_tiles, kGridX);
    uint32_t wg_y = (total_tiles + kGridX - 1) / kGridX;
    if (wg_y > 1 && wg_x != kGridX) {
      throw std::runtime_error(
          "[WebGPU QuantizedMatmul] GEMV dispatch invariant violated.");
    }
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y, batch_size);
  } else {
    uint32_t output_size = M * N;
    uint32_t num_workgroups_x =
        (output_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
    constexpr uint32_t kGridX = 65535u;
    uint32_t wg_x = std::min(num_workgroups_x, kGridX);
    uint32_t wg_y = (num_workgroups_x + kGridX - 1) / kGridX;
    if (wg_y > kGridX) {
      throw std::runtime_error(
          "[WebGPU QuantizedMatmul] Output is too large for a 2D WebGPU dispatch.");
    }
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y, batch_size);
  }

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler(
      [uniform_buf]() { wgpu::device().uniform_pool().release(uniform_buf); });
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
  auto ensure_compact_row_contiguous_matrix = [&](const array& arr) {
    if (arr.ndim() < 2) {
      if (arr.offset() == 0 && arr.flags().contiguous &&
          arr.size() == arr.data_size()) {
        return arr;
      }
      auto arr_copy = contiguous_copy_gpu(arr, s);
      encoder.add_temporary(arr_copy);
      return arr_copy;
    }
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (arr.offset() == 0 && arr.flags().contiguous &&
        arr.size() == arr.data_size() && stride_0 == arr.shape(-1) &&
        stride_1 == 1) {
      return arr;
    }
    auto arr_copy = contiguous_copy_gpu(arr, s);
    encoder.add_temporary(arr_copy);
    return arr_copy;
  };

  array x = ensure_compact_row_contiguous_matrix(inputs[0]);
  array w = ensure_compact_row_contiguous_matrix(inputs[1]);
  array scales = ensure_compact_row_contiguous_matrix(inputs[2]);
  array biases = ensure_compact_row_contiguous_matrix(inputs[3]);
  const array& lhs_indices = inputs[4];
  const array& rhs_indices = inputs[5];

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  if (out.size() == 0) {
    return;
  }

  uint32_t M = checked_u32(out.shape(-2), "[WebGPU GatherQMM]", "M");
  uint32_t N = checked_u32(out.shape(-1), "[WebGPU GatherQMM]", "N");
  uint32_t K = checked_u32(x.shape(-1), "[WebGPU GatherQMM]", "K");
  uint32_t index_size = checked_u32(
      out.size() / (static_cast<int64_t>(M) * static_cast<int64_t>(N)),
      "[WebGPU GatherQMM]",
      "index_size");
  if (lhs_indices.size() != index_size || rhs_indices.size() != index_size) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] index tensor sizes must match the output batch size.");
  }
  if (lhs_indices.ndim() != rhs_indices.ndim()) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] lhs_indices and rhs_indices must have matching rank.");
  }
  if (lhs_indices.ndim() > wgpu::MAX_NDIM) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] index rank exceeds WebGPU MAX_NDIM.");
  }
  for (int d = 0; d < lhs_indices.ndim(); ++d) {
    if (lhs_indices.shape(d) != rhs_indices.shape(d)) {
      throw std::runtime_error(
          "[WebGPU GatherQMM] lhs_indices and rhs_indices must have matching shapes.");
    }
  }
  uint32_t w_cols = checked_u32(w.shape(-1), "[WebGPU GatherQMM]", "w_cols");
  uint32_t num_groups = K / static_cast<uint32_t>(group_size_);

  const char* x_type = wgpu::dtype_to_wgsl_safe(x.dtype());
  const char* scale_type = wgpu::dtype_to_wgsl_safe(scales.dtype());
  const char* bias_type = wgpu::dtype_to_wgsl_safe(biases.dtype());
  const char* out_type = wgpu::dtype_to_wgsl_safe(out.dtype());
  std::string entry_name = std::string("gather_qmm_") + x_type + "_s" +
      scale_type + "_z" + bias_type + "_o" + out_type + "_b" +
      std::to_string(bits_) + "_g" + std::to_string(group_size_);
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(entry_name, [&]() {
    return make_gather_qmm_kernel(
        entry_name,
        x_type,
        scale_type,
        bias_type,
        out_type,
        bits_,
        group_size_);
  });
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
  params.index_ndim =
      checked_u32(lhs_indices.ndim(), "[WebGPU GatherQMM]", "index_ndim");
  params.lhs_offset = shader_elem_offset(lhs_indices, "[WebGPU GatherQMM]");
  params.rhs_offset = shader_elem_offset(rhs_indices, "[WebGPU GatherQMM]");
  params.index_shape.fill(1);
  params.lhs_strides.fill(0);
  params.rhs_strides.fill(0);
  for (int d = 0; d < lhs_indices.ndim(); ++d) {
    params.index_shape[d] =
        checked_u32(lhs_indices.shape(d), "[WebGPU GatherQMM]", "index_shape");
    params.lhs_strides[d] = checked_i32(
        lhs_indices.strides()[d], "[WebGPU GatherQMM]", "lhs_strides");
    params.rhs_strides[d] = checked_i32(
        rhs_indices.strides()[d], "[WebGPU GatherQMM]", "rhs_strides");
  }

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf = pool.acquire(
      wgpu::device().gpu_queue(), &params, sizeof(GatherQMMParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases);
  WGPUBuffer lhs_buf = wgpu::wgpu_buffer(lhs_indices);
  WGPUBuffer rhs_buf = wgpu::wgpu_buffer(rhs_indices);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{x_buf, wgpu::wgpu_bind_size(x)},
       {w_buf, wgpu::wgpu_bind_size(w)},
       {s_buf, wgpu::wgpu_bind_size(scales)},
       {b_buf, wgpu::wgpu_bind_size(biases)},
       {lhs_buf, wgpu::wgpu_bind_size(lhs_indices)},
       {rhs_buf, wgpu::wgpu_bind_size(rhs_indices)},
       {out_buf, wgpu::wgpu_bind_size(out)},
       {uniform_buf, sizeof(GatherQMMParams)}});

  uint32_t output_size =
      checked_u32(out.size(), "[WebGPU GatherQMM]", "output_size");
  uint32_t num_workgroups =
      (output_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  constexpr uint32_t kGridX = 65535u;
  uint32_t wg_x = std::min(num_workgroups, kGridX);
  uint32_t wg_y = (num_workgroups + kGridX - 1) / kGridX;
  if (wg_y > kGridX) {
    throw std::runtime_error(
        "[WebGPU GatherQMM] Output is too large for a 2D WebGPU dispatch.");
  }
  encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y, 1);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler(
      [uniform_buf]() { wgpu::device().uniform_pool().release(uniform_buf); });
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
    if (arr.offset() == 0 && arr.flags().contiguous) {
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
  const char* scale_type = wgpu::dtype_to_wgsl_safe(scales.dtype());
  const char* bias_type = wgpu::dtype_to_wgsl_safe(biases.dtype());
  std::string entry_name = std::string("affine_dequantize_") + out_type + "_s" +
      scale_type + "_z" + bias_type + "_b" + std::to_string(bits_) + "_g" +
      std::to_string(group_size_);
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(entry_name, [&]() {
    return make_affine_dequantize_kernel(
        entry_name, out_type, scale_type, bias_type, bits_, group_size_);
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
  WGPUBuffer uniform_buf = pool.acquire(
      wgpu::device().gpu_queue(), &params, sizeof(DequantizeParams));

  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer s_buf = wgpu::wgpu_buffer(scales);
  WGPUBuffer b_buf = wgpu::wgpu_buffer(biases);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{w_buf, wgpu::wgpu_bind_size(w)},
       {s_buf, wgpu::wgpu_bind_size(scales)},
       {b_buf, wgpu::wgpu_bind_size(biases)},
       {out_buf, wgpu::wgpu_bind_size(out)},
       {uniform_buf, sizeof(DequantizeParams)}});

  encoder.dispatch_compute(pe.pipeline, bg, grid_x, grid_y, 1);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler(
      [uniform_buf]() { wgpu::device().uniform_pool().release(uniform_buf); });
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[WebGPU] QQMatmul is not implemented. "
      "This operation is rarely used.");
}

} // namespace mlx::core
