// Copyright 2026 Apple Inc.
//
// WebGPU LogSumExp implementation.
//
// LogSumExp(x) = log(sum(exp(x - max(x)))) + max(x)
//
// Uses the same two-pass reduction pattern as softmax:
//   1. Compute max of the last axis (parallel workgroup reduction)
//   2. Compute sum of exp(x - max) (parallel workgroup reduction)
//   3. Write log(sum) + max to the single output element per row
//
// One workgroup (256 threads) per row. For rows longer than 256, each thread
// handles multiple elements in a loop.

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
// Uniform params (vec4-aligned)
// ---------------------------------------------------------------------------

struct LogSumExpParams {
  uint32_t data[4]; // [axis_size, pad, pad, pad]
};

// ---------------------------------------------------------------------------
// WGSL kernel generation
// ---------------------------------------------------------------------------

std::string make_logsumexp_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& acc_type) {
  std::ostringstream s;

  if (in_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct LogSumExpParams {\n"
    << "  data: vec4<u32>,\n"
    << "}\n\n";

  // Input is the full [..., axis_size] array; output has one element per row.
  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << in_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: LogSumExpParams;\n\n";

  s << "var<workgroup> shared_max: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n";
  s << "var<workgroup> shared_sum: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n\n";

  s << "fn sum_op(a: " << acc_type << ", b: " << acc_type << ") -> "
    << acc_type << " { return a + b; }\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let axis_size = params.data.x;\n"
    << "  let tid = lid.x;\n"
    << "  let row = wg_id.x;\n"
    << "  let row_start = row * axis_size;\n"
    << "\n"
    << "  // Phase 1: find max over the row\n"
    << "  var thread_max: " << acc_type << " = " << acc_type
    << "(-3.402823e+38);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n"
    << "    thread_max = max(thread_max, val);\n"
    << "  }\n"
    << "\n"
    << "  shared_max[tid] = thread_max;\n"
    << "  workgroupBarrier();\n\n";

  wgpu::emit_unrolled_reduction(s, "shared_max", "max");

  s << "  let row_max: " << acc_type << " = shared_max[0];\n"
    << "\n"
    << "  // Phase 2: sum of exp(x - max)\n"
    << "  var thread_sum: " << acc_type << " = " << acc_type << "(0.0);\n"
    << "  for (var i: u32 = tid; i < axis_size; i = i + WORKGROUP_SIZE) {\n"
    << "    let val = " << acc_type << "(input[row_start + i]);\n"
    << "    thread_sum = thread_sum + exp(val - row_max);\n"
    << "  }\n"
    << "\n"
    << "  shared_sum[tid] = thread_sum;\n"
    << "  workgroupBarrier();\n\n";

  wgpu::emit_unrolled_reduction(s, "shared_sum", "sum_op");

  // Write log(sum) + max. Handle inf max (all-inf input) to match CPU behavior.
  s << "  if (tid == 0u) {\n"
    << "    let s = shared_sum[0];\n"
    << "    if (row_max >= " << acc_type << "(3.402823e+38) || row_max <= "
    << acc_type << "(-3.402823e+38)) {\n"
    << "      output[row] = " << in_type << "(row_max);\n"
    << "    } else {\n"
    << "      output[row] = " << in_type << "(log(s) + row_max);\n"
    << "    }\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// LogSumExp::eval_gpu
// ---------------------------------------------------------------------------

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& s = stream();

  const auto& in_orig = inputs[0];

  // RED-F016: scalar input has no "last axis" — indexing strides()/shape()
  // below would be UB on a 0-D array. logsumexp of a single scalar is
  // degenerate; reject with a clear error rather than silently reading past
  // the end of empty vectors.
  if (in_orig.ndim() == 0) {
    throw std::invalid_argument(
        "[LogSumExp::eval_gpu] LogSumExp requires a non-scalar input; got ndim==0.");
  }

  // Ensure the last dimension is contiguous (stride-1) before reduction.
  array in = in_orig;
  bool needs_copy =
      !in.flags().contiguous || in.strides()[in.ndim() - 1] != 1;
  if (needs_copy) {
    in = contiguous_copy_gpu(in_orig, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(in);
  }

  // Output shape is [..., 1]: one scalar per row.
  // The output element count equals (in.data_size() / axis_size).
  int axis_size = in.shape().back();
  int n_rows = static_cast<int>(in.data_size()) / axis_size;

  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  if (n_rows == 0 || axis_size == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string in_type(in_wgsl);
  // Always accumulate in f32 for precision.
  std::string acc_type = "f32";

  std::string entry_name = std::string("logsumexp_") + in_type;
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_logsumexp_kernel(entry_name, in_type, acc_type);
      });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  LogSumExpParams params{};
  params.data[0] = static_cast<uint32_t>(axis_size);

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf = pool.acquire(
      wgpu::device().gpu_queue(), &params, sizeof(LogSumExpParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpu::wgpu_bind_size(in);
  uint64_t out_buf_size = wgpu::wgpu_bind_size(out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(LogSumExpParams)}});

  // One workgroup per row.
  encoder.dispatch_compute(pe.pipeline, bg, static_cast<uint32_t>(n_rows));

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core
