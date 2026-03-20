// Copyright 2026 Apple Inc.
//
// WebGPU softmax implementation.
//
// Uses online softmax with 3 phases in a single workgroup:
//   1. Compute max of input row (parallel reduction)
//   2. Compute sum of exp(x - max) (parallel reduction)
//   3. Output exp(x - max) / sum (parallel apply)
//
// One workgroup (256 threads) per row. For rows longer than 256,
// each thread handles multiple elements in a loop.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// ---------------------------------------------------------------------------
// Softmax params
// ---------------------------------------------------------------------------

struct SoftmaxParams {
  uint32_t data[4]; // [axis_size, pad, pad, pad]
};

// ---------------------------------------------------------------------------
// WGSL kernel generation: softmax
// ---------------------------------------------------------------------------

std::string make_softmax_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& acc_type) {
  std::ostringstream s;

  if (in_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct SoftmaxParams {\n"
    << "  data: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << in_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: SoftmaxParams;\n\n";

  s << "var<workgroup> shared_max: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n";
  s << "var<workgroup> shared_sum: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n\n";

  // Generate the kernel
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let axis_size = params.data.x;\n"
    << "  let tid = lid.x;\n"
    << "  let row = wg_id.x;\n"
    << "  let row_start = row * axis_size;\n"
    << "\n"
    << "  // Phase 1: Compute max via parallel reduction\n"
    << "  var thread_max: " << acc_type << " = " << acc_type
    << "(-3.402823e+38);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n"
    << "    thread_max = max(thread_max, val);\n"
    << "  }\n"
    << "\n"
    << "  // Store to shared memory and reduce\n"
    << "  shared_max[tid] = thread_max;\n"
    << "  workgroupBarrier();\n"
    << "\n"
    << "  for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {\n"
    << "    if (tid < stride) {\n"
    << "      shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "  }\n"
    << "  let row_max = shared_max[0];\n"
    << "\n"
    << "  // Phase 2: Compute sum of exp(x - max)\n"
    << "  var thread_sum: " << acc_type << " = " << acc_type << "(0.0);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n"
    << "    thread_sum = thread_sum + exp(val - row_max);\n"
    << "  }\n"
    << "\n"
    << "  shared_sum[tid] = thread_sum;\n"
    << "  workgroupBarrier();\n"
    << "\n"
    << "  for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {\n"
    << "    if (tid < stride) {\n"
    << "      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];\n"
    << "    }\n"
    << "    workgroupBarrier();\n"
    << "  }\n"
    << "  let inv_sum = " << acc_type << "(1.0) / shared_sum[0];\n"
    << "\n"
    << "  // Phase 3: Write output = exp(x - max) / sum\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n"
    << "    output[row_start + i] = " << in_type
    << "(exp(val - row_max) * inv_sum);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point: Softmax::eval_gpu
// ---------------------------------------------------------------------------

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& s = stream();

  const auto& in_orig = inputs[0];

  // Make sure the last dimension is contiguous.
  // If the input is contiguous with stride-1 last axis, we can work in-place.
  array in = in_orig;
  bool needs_copy = !in.flags().contiguous || in.strides()[in.ndim() - 1] != 1;

  if (needs_copy) {
    in = contiguous_copy_gpu(in_orig, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(in);
  }

  // Set up output. We always need a separate buffer because the WGSL shader
  // binds input as read-only and output as read-write, and WebGPU does not
  // allow the same buffer in both roles simultaneously.
  out.set_data(
      allocator::malloc(in.data_size() * in.itemsize()),
      in.data_size(),
      in.strides(),
      in.flags());

  int axis_size = in.shape().back();
  int n_rows = static_cast<int>(in.data_size()) / axis_size;

  if (n_rows == 0 || axis_size == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl(in.dtype());
  std::string in_type(in_wgsl);
  // Always accumulate in f32 for precision
  std::string acc_type = "f32";

  std::string entry_name = std::string("softmax_") + in_type;
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_softmax_kernel(entry_name, in_type, acc_type); });
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  SoftmaxParams params{};
  params.data[0] = static_cast<uint32_t>(axis_size);

  WGPUBuffer uniform_buf =
      wgpu::create_uniform_buffer(&params, sizeof(SoftmaxParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pipeline,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(SoftmaxParams)}});

  // One workgroup per row
  encoder.dispatch_compute(
      pipeline, bg, static_cast<uint32_t>(n_rows));

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
}

} // namespace mlx::core
