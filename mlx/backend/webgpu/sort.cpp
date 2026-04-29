// Copyright 2026 Apple Inc.
//
// WebGPU Sort and ArgSort implementation.
//
// Uses bitonic sort (O(n log^2 n) comparisons, GPU-friendly).
//
// Strategy:
//   - One workgroup per "row" (slice along the sort axis).
//   - Each thread handles multiple elements (axis_size / WORKGROUP_SIZE).
//   - Data is loaded into workgroup shared memory, sorted there, then written
//     back.
//   - For Sort: only keys are stored in shared memory.
//   - For ArgSort: both keys and index values are stored.
//
// Shared memory budget: WebGPU guarantees at least 16384 bytes.
//   - Sort (keys only, f32): 4096 * 4 = 16384 bytes -> max 4096 elements
//   - ArgSort (keys + vals): 2048 * (4+4) = 16384 bytes -> max 2048 elements
//
// The sort axis is moved to the last dimension via contiguous copy if needed,
// so the kernel always sorts along the innermost (contiguous) axis.

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
// Shared memory limits
// ---------------------------------------------------------------------------

// WebGPU guarantees at least 16384 bytes of workgroup storage.
// For Sort (keys only, f32 = 4 bytes): 16384 / 4 = 4096 elements max
// For ArgSort (keys f32 + vals u32 = 8 bytes): 16384 / 8 = 2048 elements max
// We use conservative limits that work for all supported dtypes.
constexpr uint32_t SORT_MAX_AXIS = 4096;
constexpr uint32_t ARGSORT_MAX_AXIS = 2048;

constexpr uint32_t SORT_ELEMS_PER_THREAD =
    SORT_MAX_AXIS / wgpu::WORKGROUP_SIZE;     // 16
constexpr uint32_t ARGSORT_ELEMS_PER_THREAD =
    ARGSORT_MAX_AXIS / wgpu::WORKGROUP_SIZE;  // 8

// ---------------------------------------------------------------------------
// Uniform params (vec4-aligned)
// ---------------------------------------------------------------------------

struct SortParams {
  uint32_t data[4]; // [axis_size, n_padded, pad, pad]
};

// ---------------------------------------------------------------------------
// WGSL kernel generation: bitonic sort
// ---------------------------------------------------------------------------

// Generate a bitonic sort kernel that sorts along the last (contiguous) axis.
// Each workgroup processes one row. The kernel works for any axis_size up to
// the shared memory limit by padding to the next power of 2 and using sentinel
// values.
//
// For Sort: output is sorted values (same type as input).
// For ArgSort: output is uint32 indices into the original row.

std::string make_sort_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    bool argsort) {
  uint32_t max_elems = argsort ? ARGSORT_MAX_AXIS : SORT_MAX_AXIS;
  uint32_t elems_per_thread = argsort ? ARGSORT_ELEMS_PER_THREAD
                                      : SORT_ELEMS_PER_THREAD;

  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n";
  s << "const ELEMS_PER_THREAD: u32 = " << elems_per_thread << "u;\n\n";

  s << "struct SortParams {\n"
    << "  data: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << val_type
    << ">;\n";
  if (argsort) {
    s << "@group(0) @binding(1) var<storage, read_write> output: array<u32>;\n";
  } else {
    s << "@group(0) @binding(1) var<storage, read_write> output: array<"
      << val_type << ">;\n";
  }
  s << "@group(0) @binding(2) var<uniform> params: SortParams;\n\n";

  // Shared memory: keys always needed, vals only for argsort
  s << "var<workgroup> sh_keys: array<" << val_type << ", " << max_elems
    << ">;\n";
  if (argsort) {
    s << "var<workgroup> sh_vals: array<u32, " << max_elems << ">;\n";
  }
  s << "\n";

  // Sentinel value for padding (sorts to the end for ascending order)
  std::string sentinel;
  if (val_type == "f32") {
    sentinel = "f32(3.402823e+38)";
  } else if (val_type == "f16") {
    sentinel = "f16(65504.0)";
  } else if (val_type == "i32") {
    sentinel = "i32(2147483647)";
  } else if (val_type == "u32") {
    sentinel = "u32(4294967295u)";
  } else {
    sentinel = val_type + "(3.402823e+38)";
  }

  // The kernel
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let axis_size = params.data.x;\n"
    << "  let n_padded = params.data.y;\n"
    << "  let tid = lid.x;\n"
    << "  let row = wg_id.x;\n"
    << "  let row_start = row * axis_size;\n"
    << "\n";

  // Load phase: each thread loads elems_per_thread elements
  // Thread t loads indices [t*elems_per_thread, (t+1)*elems_per_thread)
  s << "  // Load data into shared memory\n"
    << "  for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {\n"
    << "    let idx = tid * ELEMS_PER_THREAD + e;\n"
    << "    if (idx < n_padded) {\n"
    << "      if (idx < axis_size) {\n"
    << "        sh_keys[idx] = input[row_start + idx];\n";
  if (argsort) {
    s << "        sh_vals[idx] = idx;\n";
  }
  s << "      } else {\n"
    << "        sh_keys[idx] = " << sentinel << ";\n";
  if (argsort) {
    s << "        sh_vals[idx] = idx;\n";
  }
  s << "      }\n"
    << "    }\n"
    << "  }\n"
    << "  workgroupBarrier();\n\n";

  // Bitonic sort network
  // Outer loop: k = 2, 4, 8, ..., n_padded (stage size)
  // Inner loop: j = k/2, k/4, ..., 1 (compare distance)
  // For each (k, j), each element t is paired with t XOR j.
  // Sort direction: ascending if (t & k) == 0.
  // Only the element with the lower index in the pair does the compare-swap.
  s << "  // Bitonic sort\n"
    << "  var k: u32 = 2u;\n"
    << "  loop {\n"
    << "    if (k > n_padded) { break; }\n"
    << "    var j: u32 = k >> 1u;\n"
    << "    loop {\n"
    << "      if (j == 0u) { break; }\n"
    << "      for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {\n"
    << "        let t = tid * ELEMS_PER_THREAD + e;\n"
    << "        if (t < n_padded) {\n"
    << "          let partner = t ^ j;\n"
    << "          if (partner > t && partner < n_padded) {\n"
    << "            let ascending = ((t & k) == 0u);\n"
    << "            let key_t = sh_keys[t];\n"
    << "            let key_p = sh_keys[partner];\n"
    << "            let should_swap = (ascending && (key_t > key_p)) || (!ascending && (key_t < key_p));\n"
    << "            if (should_swap) {\n"
    << "              sh_keys[t] = key_p;\n"
    << "              sh_keys[partner] = key_t;\n";
  if (argsort) {
    s << "              let tmp_v = sh_vals[t];\n"
      << "              sh_vals[t] = sh_vals[partner];\n"
      << "              sh_vals[partner] = tmp_v;\n";
  }
  s << "            }\n"
    << "          }\n"
    << "        }\n"
    << "      }\n"
    << "      workgroupBarrier();\n"
    << "      j = j >> 1u;\n"
    << "    }\n"
    << "    k = k << 1u;\n"
    << "  }\n\n";

  // Write results back
  s << "  // Write sorted results\n"
    << "  for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {\n"
    << "    let idx = tid * ELEMS_PER_THREAD + e;\n"
    << "    if (idx < axis_size) {\n";
  if (argsort) {
    s << "      output[row * axis_size + idx] = sh_vals[idx];\n";
  } else {
    s << "      output[row * axis_size + idx] = sh_keys[idx];\n";
  }
  s << "    }\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// Next power of 2 >= n
uint32_t next_power_of_2(uint32_t n) {
  if (n <= 1)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// ---------------------------------------------------------------------------
// Common eval logic for Sort and ArgSort
// ---------------------------------------------------------------------------

void eval_sort_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis_,
    bool argsort) {
  assert(inputs.size() == 1);
  auto& s_stream = out.primitive().stream();

  const auto& in_orig = inputs[0];
  int ndim = in_orig.ndim();

  // Resolve negative axis
  int axis = axis_ < 0 ? axis_ + ndim : axis_;

  uint32_t max_axis = argsort ? ARGSORT_MAX_AXIS : SORT_MAX_AXIS;
  uint32_t axis_size = static_cast<uint32_t>(in_orig.shape(axis));

  if (axis_size > max_axis) {
    throw std::runtime_error(
        "[Sort::eval_gpu] WebGPU bitonic sort only supports axis_size <= " +
        std::to_string(max_axis) +
        ", got " + std::to_string(axis_size) +
        ". Please use CPU fallback for larger sorts.");
  }

  // We need the sort axis to be contiguous (stride 1) and be the last axis.
  // If the sort axis is not the last one, we transpose so it becomes last,
  // then do a contiguous copy so the data layout is [n_rows, axis_size].
  array in = in_orig;
  auto& encoder = wgpu::get_command_encoder(s_stream);

  bool axis_is_last = (axis == ndim - 1);
  bool contiguous_last = in.flags().contiguous &&
      in.offset() == 0 &&
      in.size() == in.data_size() &&
      (ndim == 0 || in.strides()[ndim - 1] == 1);

  if (!axis_is_last || !contiguous_last) {
    if (!axis_is_last) {
      in = swapaxes_in_eval(in, axis, ndim - 1);
    }
    in = contiguous_copy_gpu(in, s_stream);
    encoder.add_temporary(in);
  }

  // Now 'in' has shape [..., axis_size] with stride-1 last axis.
  int n_rows = axis_size == 0
      ? 0
      : static_cast<int>(in.size()) / static_cast<int>(axis_size);

  // Allocate output
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  if (n_rows == 0 || axis_size == 0) {
    return;
  }

  auto& dev = wgpu::device();

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string val_type(in_wgsl);

  // Compute padded size (next power of 2)
  uint32_t n_padded = next_power_of_2(axis_size);

  // Build kernel
  std::string op_name = argsort ? "argsort" : "sort";
  std::string entry_name = op_name + "_" + val_type;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_sort_kernel(entry_name, val_type, argsort); });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Set params
  SortParams params{};
  params.data[0] = axis_size;
  params.data[1] = n_padded;

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(SortParams));

  // If we transposed, we need a temporary output buffer with the transposed
  // layout, then copy back to the actual output with the original axis order.
  bool need_reorder = !axis_is_last;

  array sort_out = out;
  if (need_reorder) {
    auto tmp_shape = in.shape();
    sort_out = array(tmp_shape, out.dtype(), nullptr, {});
    sort_out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(sort_out)));
    encoder.add_temporary(sort_out);
  }

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(sort_out);
  uint64_t in_buf_size = wgpu::wgpu_bind_size(in);
  uint64_t out_buf_size = wgpu::wgpu_bind_size(sort_out);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(SortParams)}});

  // One workgroup per row
  encoder.dispatch_compute(pe.pipeline, bg, static_cast<uint32_t>(n_rows));

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });

  // If we transposed, copy from temp back to out with the original layout.
  if (need_reorder) {
    // sort_out has shape = swapaxes(out.shape(), axis, ndim-1), row-contiguous.
    // Build strides for sort_out in row-major order.
    Strides sort_out_strides(ndim);
    int64_t stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
      sort_out_strides[d] = stride;
      stride *= in.shape(d);
    }
    // Swap back to match out's axis order.
    std::swap(sort_out_strides[axis], sort_out_strides[ndim - 1]);

    copy_gpu_inplace(
        sort_out,
        out,
        out.shape(),
        sort_out_strides,
        out.strides(),
        0,
        0,
        CopyType::General,
        s_stream);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Sort::eval_gpu
// ---------------------------------------------------------------------------

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_sort_gpu(inputs, out, axis_, /* argsort = */ false);
}

// ---------------------------------------------------------------------------
// ArgPartition::eval_gpu
// ---------------------------------------------------------------------------

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // WebGPU does not have a dedicated partition kernel yet. A full argsort is
  // semantically valid for argpartition users because the requested partition
  // slice still contains the same ordered side of the axis. Qwen MoE only
  // needs top-k over <=256 experts, so this keeps the path on-GPU without
  // changing native backends.
  eval_sort_gpu(inputs, out, axis_, /* argsort = */ true);
}

// ---------------------------------------------------------------------------
// ArgSort::eval_gpu
// ---------------------------------------------------------------------------

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_sort_gpu(inputs, out, axis_, /* argsort = */ true);
}

} // namespace mlx::core
