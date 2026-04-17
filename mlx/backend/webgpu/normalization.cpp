// Copyright 2026 Apple Inc.
//
// WebGPU fused RMSNorm and LayerNorm implementations.
//
// RMSNorm uses 2 phases in a single workgroup:
//   1. Compute sum of squares via parallel reduction -> inverseSqrt
//   2. Output = input * inv_rms * weight
//
// LayerNorm uses 3 phases in a single workgroup:
//   1. Compute mean via parallel reduction
//   2. Compute variance via parallel reduction (using x - mean)
//   3. Output = (x - mean) * inv_std * weight + bias
//
// One workgroup (256 threads) per row. For rows longer than 256,
// each thread handles multiple elements in a loop.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/fast_primitives.h"

#include <cassert>
#include <sstream>

#ifdef MLX_WGPU_LOG_KERNELS
#include <iostream>
#endif

namespace mlx::core::fast {

namespace {

// ---------------------------------------------------------------------------
// RMSNorm params — must match the WGSL struct layout
// ---------------------------------------------------------------------------

struct RMSNormParams {
  uint32_t data[4]; // [axis_size, w_stride, eps_bits, pad]
};

// ---------------------------------------------------------------------------
// LayerNorm params — must match the WGSL struct layout
// ---------------------------------------------------------------------------

struct LayerNormParams {
  uint32_t data[4]; // [axis_size, w_stride, eps_bits, has_bias]
};

// ---------------------------------------------------------------------------
// WGSL kernel generation: rmsnorm
// ---------------------------------------------------------------------------

std::string make_rmsnorm_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& w_type,
    const std::string& acc_type,
    int subgroup_size,
    bool w_packed_bf16) {
  const bool use_subgroups = subgroup_size > 0;
  std::ostringstream s;

  if (use_subgroups) {
    s << "enable subgroups;\n";
  }
  if (in_type == "f16" || w_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct RMSNormParams {\n"
    << "  data: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n";
  if (w_packed_bf16) {
    // Packed bf16: two bf16 values per u32 slot. Kernel unpacks per load.
    s << "@group(0) @binding(1) var<storage, read> weight: array<u32>;\n";
  } else {
    s << "@group(0) @binding(1) var<storage, read> weight: array<" << w_type
      << ">;\n";
  }
  s << "@group(0) @binding(2) var<storage, read_write> output: array<"
    << in_type << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: RMSNormParams;\n\n";

  if (w_packed_bf16) {
    s << wgpu::wgsl_unpack_bf16_pair();
  }

  s << "var<workgroup> shared_sum: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n\n";

  // Helper for tree reduction
  s << "fn sum_op(a: " << acc_type << ", b: " << acc_type << ") -> "
    << acc_type << " { return a + b; }\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let axis_size = params.data.x;\n"
    << "  let w_stride = params.data.y;\n"
    << "  let eps = bitcast<f32>(params.data.z);\n"
    << "  let tid = lid.x;\n"
    << "  let row = wg_id.x;\n"
    << "  let row_start = row * axis_size;\n"
    << "\n"
    << "  // Phase 1: Compute sum of squares via parallel reduction\n"
    << "  var thread_sum: " << acc_type << " = " << acc_type << "(0.0);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n"
    << "    thread_sum = thread_sum + val * val;\n"
    << "  }\n"
    << "\n";

  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s, "thread_sum", "shared_sum", "subgroupAdd", "sum_op",
        wgpu::WORKGROUP_SIZE, static_cast<uint32_t>(subgroup_size));
  } else {
    s << "  shared_sum[tid] = thread_sum;\n"
      << "  workgroupBarrier();\n\n";
    wgpu::emit_unrolled_reduction(s, "shared_sum", "sum_op");
  }

  s << "  let mean_sq = shared_sum[0] / " << acc_type << "(axis_size);\n"
    << "  let inv_rms = inverseSqrt(mean_sq + " << acc_type << "(eps));\n"
    << "\n"
    << "  // Phase 2: Write output = input * inv_rms * weight\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n";
  if (w_packed_bf16) {
    // w_stride is 0 (broadcast scalar) or 1 (1D contiguous). The packed
    // indexing formula works for both: w_stride=0 always reads slot 0's
    // low half (the broadcast scalar). When w_stride=1 we read element
    // `i` as `weight[i>>1].{x|y}` depending on parity.
    s << "    let w_elem_idx = i * w_stride;\n"
      << "    let w_u32_idx = w_elem_idx >> 1u;\n"
      << "    let w_parity = w_elem_idx & 1u;\n"
      << "    let w_pair = unpack_bf16_pair(weight[w_u32_idx]);\n"
      << "    let w = select(w_pair.y, w_pair.x, w_parity == 0u);\n";
  } else {
    s << "    let w = " << acc_type << "(weight[i * w_stride]);\n";
  }
  s << "    output[row_start + i] = " << in_type
    << "(val * inv_rms * w);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: layernorm
// ---------------------------------------------------------------------------

std::string make_layernorm_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& w_type,
    const std::string& acc_type,
    bool has_bias,
    int subgroup_size,
    bool w_packed_bf16,
    bool b_packed_bf16) {
  const bool use_subgroups = subgroup_size > 0;
  std::ostringstream s;

  if (use_subgroups) {
    s << "enable subgroups;\n";
  }
  if (in_type == "f16" || w_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct LayerNormParams {\n"
    << "  data: vec4<u32>,\n"
    << "}\n\n";

  // Bindings: input, weight, [bias], output, params
  uint32_t binding = 0;
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> input: array<"
    << in_type << ">;\n";
  if (w_packed_bf16) {
    s << "@group(0) @binding(" << binding++
      << ") var<storage, read> weight: array<u32>;\n";
  } else {
    s << "@group(0) @binding(" << binding++
      << ") var<storage, read> weight: array<" << w_type << ">;\n";
  }
  if (has_bias) {
    if (b_packed_bf16) {
      s << "@group(0) @binding(" << binding++
        << ") var<storage, read> bias: array<u32>;\n";
    } else {
      s << "@group(0) @binding(" << binding++
        << ") var<storage, read> bias: array<" << w_type << ">;\n";
    }
  }
  s << "@group(0) @binding(" << binding++ << ") var<storage, read_write> output: array<"
    << in_type << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<uniform> params: LayerNormParams;\n\n";

  if (w_packed_bf16 || b_packed_bf16) {
    s << wgpu::wgsl_unpack_bf16_pair();
  }

  // Shared memory for reductions (reused for mean and variance)
  s << "var<workgroup> shared_sum: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n\n";

  // Helper for tree reduction
  s << "fn sum_op(a: " << acc_type << ", b: " << acc_type << ") -> "
    << acc_type << " { return a + b; }\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let axis_size = params.data.x;\n"
    << "  let w_stride = params.data.y;\n"
    << "  let eps = bitcast<f32>(params.data.z);\n"
    << "  let tid = lid.x;\n"
    << "  let row = wg_id.x;\n"
    << "  let row_start = row * axis_size;\n"
    << "\n";

  // Phase 1: Compute mean via parallel reduction
  s << "  // Phase 1: Compute mean via parallel reduction\n"
    << "  var thread_sum: " << acc_type << " = " << acc_type << "(0.0);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    thread_sum = thread_sum + " << acc_type << "(input[row_start + i]);\n"
    << "  }\n"
    << "\n";

  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s, "thread_sum", "shared_sum", "subgroupAdd", "sum_op",
        wgpu::WORKGROUP_SIZE, static_cast<uint32_t>(subgroup_size), "mn");
  } else {
    s << "  shared_sum[tid] = thread_sum;\n"
      << "  workgroupBarrier();\n\n";
    wgpu::emit_unrolled_reduction(s, "shared_sum", "sum_op");
  }

  s << "  let mean_val = shared_sum[0] / " << acc_type << "(axis_size);\n"
    << "\n";

  // Phase 2: Compute variance via parallel reduction
  s << "  // Phase 2: Compute variance via parallel reduction\n"
    << "  var thread_var: " << acc_type << " = " << acc_type << "(0.0);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let diff = " << acc_type << "(input[row_start + i]) - mean_val;\n"
    << "    thread_var = thread_var + diff * diff;\n"
    << "  }\n"
    << "\n";

  // Need a barrier before reusing shared_sum
  s << "  workgroupBarrier();\n\n";

  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s, "thread_var", "shared_sum", "subgroupAdd", "sum_op",
        wgpu::WORKGROUP_SIZE, static_cast<uint32_t>(subgroup_size), "vr");
  } else {
    s << "  shared_sum[tid] = thread_var;\n"
      << "  workgroupBarrier();\n\n";
    wgpu::emit_unrolled_reduction(s, "shared_sum", "sum_op");
  }

  s << "  let inv_std = inverseSqrt(shared_sum[0] / " << acc_type
    << "(axis_size) + " << acc_type << "(eps));\n"
    << "\n";

  // Phase 3: Output = (x - mean) * inv_std * weight + bias
  s << "  // Phase 3: Output = (x - mean) * inv_std * weight + bias\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = (" << acc_type << "(input[row_start + i]) - mean_val) * inv_std;\n";
  if (w_packed_bf16) {
    // w_stride is 0 (broadcast) or 1 (1D contiguous); packed formula
    // handles both — broadcast always reads slot 0 low half.
    s << "    let w_elem_idx = i * w_stride;\n"
      << "    let w_u32_idx = w_elem_idx >> 1u;\n"
      << "    let w_parity = w_elem_idx & 1u;\n"
      << "    let w_pair = unpack_bf16_pair(weight[w_u32_idx]);\n"
      << "    let w = select(w_pair.y, w_pair.x, w_parity == 0u);\n";
  } else {
    s << "    let w = " << acc_type << "(weight[i * w_stride]);\n";
  }

  if (has_bias) {
    if (b_packed_bf16) {
      s << "    let b_elem_idx = i * w_stride;\n"
        << "    let b_u32_idx = b_elem_idx >> 1u;\n"
        << "    let b_parity = b_elem_idx & 1u;\n"
        << "    let b_pair = unpack_bf16_pair(bias[b_u32_idx]);\n"
        << "    let b = select(b_pair.y, b_pair.x, b_parity == 0u);\n";
    } else {
      s << "    let b = " << acc_type << "(bias[i * w_stride]);\n";
    }
    s << "    output[row_start + i] = " << in_type << "(val * w + b);\n";
  } else {
    s << "    output[row_start + i] = " << in_type << "(val * w);\n";
  }

  s << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point: fast::RMSNorm::eval_gpu
// ---------------------------------------------------------------------------

bool RMSNorm::use_fallback(Stream s) {
  return false;  // GPU implementation
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& s = stream();
  auto& out = outputs[0];

  const auto& x_orig = inputs[0];
  const auto& w = inputs[1];

  // Ensure input is row-contiguous with zero offset.
  // WebGPU kernels always bind buffers from offset 0, so any non-zero
  // array offset would cause the kernel to read wrong data.
  array x = x_orig;
  bool needs_copy = !x.flags().row_contiguous || x.offset() != 0;

  if (needs_copy) {
    x = contiguous_copy_gpu(x_orig, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(x);
  }

  // Allocate output — separate buffer required for WebGPU
  out.set_data(
      allocator::malloc(wgpu::wgpu_alloc_size(out)),
      x.data_size(),
      x.strides(),
      x.flags());

  int axis_size = x.shape().back();
  int n_rows = static_cast<int>(x.size()) / axis_size;

  if (n_rows == 0 || axis_size == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(x.dtype());
  const char* w_wgsl = wgpu::dtype_to_wgsl_safe(w.dtype());
  std::string in_type(in_wgsl);
  std::string w_type(w_wgsl);
  std::string acc_type = "f32"; // Always accumulate in f32

  int sg = wgpu::effective_subgroup_size();
  std::string sg_suffix = wgpu::subgroup_suffix();

  // Detect whether the weight buffer is stored in packed-bf16 layout
  // (2 bf16 values per u32 slot). The Step C norm-weight allowlist in
  // gpu-worker.ts can flip 1-D bf16 norm weights into PackedBf16 mode on
  // first upload — the kernel then binds `weight` as `array<u32>` and
  // unpacks per element. Safe only when the source dtype is bf16; any
  // other weight type means the buffer was never packed.
  auto* w_wbuf = static_cast<const wgpu::WebGPUBuffer*>(w.buffer().ptr());
  const bool w_packed_bf16 = (w.dtype() == bfloat16) && (w_wbuf != nullptr) &&
      (w_wbuf->storage_mode == wgpu::StorageMode::PackedBf16);

  std::string packed_suffix = w_packed_bf16 ? "_bf16p" : "";
  std::string entry_name = std::string("rmsnorm_") + in_type + "_w" + w_type +
      packed_suffix + sg_suffix;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
#ifdef MLX_WGPU_LOG_KERNELS
        // First-dispatch log — the shader-module cache only invokes this
        // lambda on a MISS, so each unique entry_name prints exactly once.
        std::cerr << "[MLX-KERNEL] " << entry_name << std::endl;
#endif
        return make_rmsnorm_kernel(
            entry_name, in_type, w_type, acc_type, sg, w_packed_bf16);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);

  RMSNormParams params{};
  params.data[0] = static_cast<uint32_t>(axis_size);
  params.data[1] = (w.ndim() == 1) ? static_cast<uint32_t>(w.strides()[0]) : 0;
  float eps_val = eps_;
  uint32_t eps_bits;
  std::memcpy(&eps_bits, &eps_val, sizeof(eps_bits));
  params.data[2] = eps_bits;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(RMSNormParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t x_buf_size = wgpuBufferGetSize(x_buf);
  uint64_t w_buf_size = wgpuBufferGetSize(w_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{x_buf, x_buf_size},
       {w_buf, w_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(RMSNormParams)}});

  // One workgroup per row
  encoder.dispatch_compute(
      pe.pipeline, bg, static_cast<uint32_t>(n_rows));

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

// ---------------------------------------------------------------------------
// Main entry point: fast::LayerNorm::eval_gpu
// ---------------------------------------------------------------------------

bool LayerNorm::use_fallback(Stream s) {
  return false;  // GPU implementation
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() >= 2 && inputs.size() <= 3);
  auto& s = stream();
  auto& out = outputs[0];

  const auto& x_orig = inputs[0];
  const auto& w = inputs[1];
  bool has_bias = (inputs.size() == 3);

  // Ensure input is row-contiguous with zero offset.
  // WebGPU kernels always bind buffers from offset 0, so any non-zero
  // array offset would cause the kernel to read wrong data.
  array x = x_orig;
  bool needs_copy = !x.flags().row_contiguous || x.offset() != 0;

  if (needs_copy) {
    x = contiguous_copy_gpu(x_orig, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(x);
  }

  // Allocate output — separate buffer required for WebGPU
  out.set_data(
      allocator::malloc(wgpu::wgpu_alloc_size(out)),
      x.data_size(),
      x.strides(),
      x.flags());

  int axis_size = x.shape().back();
  int n_rows = static_cast<int>(x.size()) / axis_size;

  if (n_rows == 0 || axis_size == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(x.dtype());
  const char* w_wgsl = wgpu::dtype_to_wgsl_safe(w.dtype());
  std::string in_type(in_wgsl);
  std::string w_type(w_wgsl);
  std::string acc_type = "f32"; // Always accumulate in f32

  int sg = wgpu::effective_subgroup_size();
  std::string sg_suffix = wgpu::subgroup_suffix();
  std::string bias_suffix = has_bias ? "_bias" : "_nobias";

  // Detect packed-bf16 storage on the weight (and optional bias) buffers.
  // Matches the RMSNorm::eval_gpu logic — only valid for bf16 source dtype.
  auto* w_wbuf = static_cast<const wgpu::WebGPUBuffer*>(w.buffer().ptr());
  const bool w_packed_bf16 = (w.dtype() == bfloat16) && (w_wbuf != nullptr) &&
      (w_wbuf->storage_mode == wgpu::StorageMode::PackedBf16);
  bool b_packed_bf16 = false;
  if (has_bias) {
    auto* b_wbuf =
        static_cast<const wgpu::WebGPUBuffer*>(inputs[2].buffer().ptr());
    b_packed_bf16 = (inputs[2].dtype() == bfloat16) && (b_wbuf != nullptr) &&
        (b_wbuf->storage_mode == wgpu::StorageMode::PackedBf16);
  }
  std::string packed_suffix;
  if (w_packed_bf16) packed_suffix += "_wbf16p";
  if (b_packed_bf16) packed_suffix += "_bbf16p";

  std::string entry_name = std::string("layernorm_") + in_type + "_w" +
      w_type + bias_suffix + packed_suffix + sg_suffix;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
#ifdef MLX_WGPU_LOG_KERNELS
        std::cerr << "[MLX-KERNEL] " << entry_name << std::endl;
#endif
        return make_layernorm_kernel(
            entry_name,
            in_type,
            w_type,
            acc_type,
            has_bias,
            sg,
            w_packed_bf16,
            b_packed_bf16);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  if (has_bias) {
    encoder.set_input_array(inputs[2]);
  }
  encoder.set_output_array(out);

  LayerNormParams params{};
  params.data[0] = static_cast<uint32_t>(axis_size);
  params.data[1] = (w.ndim() == 1) ? static_cast<uint32_t>(w.strides()[0]) : 0;
  float eps_val = eps_;
  uint32_t eps_bits;
  std::memcpy(&eps_bits, &eps_val, sizeof(eps_bits));
  params.data[2] = eps_bits;
  params.data[3] = has_bias ? 1u : 0u;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(LayerNormParams));

  WGPUBuffer x_buf = wgpu::wgpu_buffer(x);
  WGPUBuffer w_buf = wgpu::wgpu_buffer(w);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t x_buf_size = wgpuBufferGetSize(x_buf);
  uint64_t w_buf_size = wgpuBufferGetSize(w_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  // Build bind group entries: input, weight, [bias], output, uniform
  std::vector<std::pair<WGPUBuffer, uint64_t>> bindings;
  bindings.push_back({x_buf, x_buf_size});
  bindings.push_back({w_buf, w_buf_size});
  if (has_bias) {
    WGPUBuffer b_buf = wgpu::wgpu_buffer(inputs[2]);
    uint64_t b_buf_size = wgpuBufferGetSize(b_buf);
    bindings.push_back({b_buf, b_buf_size});
  }
  bindings.push_back({out_buf, out_buf_size});
  bindings.push_back({uniform_buf, sizeof(LayerNormParams)});

  WGPUBindGroup bg = wgpu::create_bind_group(pe.layout, bindings);

  // One workgroup per row
  encoder.dispatch_compute(
      pe.pipeline, bg, static_cast<uint32_t>(n_rows));

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core::fast
