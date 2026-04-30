// Copyright 2026 Apple Inc.
//
// WebGPU Sort and ArgSort implementation.
//
// Uses a shared-memory bitonic sort for rows that fit in one workgroup. Larger
// rows are sorted by workgroup-sized tiles and merged with bounded dispatches.
//
// Small-row strategy:
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
// so kernels always sort along the innermost (contiguous) axis.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <limits>
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
    SORT_MAX_AXIS / wgpu::WORKGROUP_SIZE; // 16
constexpr uint32_t ARGSORT_ELEMS_PER_THREAD =
    ARGSORT_MAX_AXIS / wgpu::WORKGROUP_SIZE; // 8
constexpr uint32_t LARGE_SORT_BLOCK_ELEMS = ARGSORT_MAX_AXIS;
constexpr uint32_t LARGE_SORT_ELEMS_PER_THREAD =
    LARGE_SORT_BLOCK_ELEMS / wgpu::WORKGROUP_SIZE; // 8

// ---------------------------------------------------------------------------
// Uniform params (vec4-aligned)
// ---------------------------------------------------------------------------

struct SortParams {
  uint32_t data[4]; // [axis_size, n_padded, num_rows, pad]
};

struct LargeSortParams {
  // data0: [axis_size, n_blocks, num_rows, run_width]
  // data1: [total_work_items, pad, pad, pad]
  uint32_t data0[4];
  uint32_t data1[4];
};

// ---------------------------------------------------------------------------
// WGSL kernel generation: bitonic sort
// ---------------------------------------------------------------------------

std::string sort_sentinel(const std::string& val_type) {
  if (val_type == "f32") {
    return "f32(3.402823e+38)";
  } else if (val_type == "f16") {
    return "f16(65504.0)";
  } else if (val_type == "i32") {
    return "i32(2147483647)";
  } else if (val_type == "u32") {
    return "u32(4294967295u)";
  }
  return val_type + "(3.402823e+38)";
}

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
  uint32_t elems_per_thread =
      argsort ? ARGSORT_ELEMS_PER_THREAD : SORT_ELEMS_PER_THREAD;

  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n";
  s << "const ELEMS_PER_THREAD: u32 = " << elems_per_thread << "u;\n\n";
  s << wgpu::wgsl_linear_workgroup_id();

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

  std::string sentinel = sort_sentinel(val_type);

  // The kernel
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let axis_size = params.data.x;\n"
    << "  let n_padded = params.data.y;\n"
    << "  let num_rows = params.data.z;\n"
    << "  let tid = lid.x;\n"
    << "  let row = linear_workgroup_id(wg_id, nwg);\n"
    << "  if (row >= num_rows) { return; }\n"
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

std::string make_large_sort_block_kernel(
    const std::string& entry_name,
    const std::string& val_type) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";
  s << "const BLOCK_ELEMS: u32 = " << LARGE_SORT_BLOCK_ELEMS << "u;\n";
  s << "const ELEMS_PER_THREAD: u32 = " << LARGE_SORT_ELEMS_PER_THREAD
    << "u;\n\n";
  s << wgpu::wgsl_linear_workgroup_id();

  s << "struct LargeSortParams {\n"
    << "  data0: vec4<u32>,\n"
    << "  data1: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << val_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> keys_out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read_write> idxs_out: array<u32>;\n"
    << "@group(0) @binding(3) var<uniform> params: LargeSortParams;\n\n";

  s << "var<workgroup> sh_keys: array<" << val_type << ", "
    << LARGE_SORT_BLOCK_ELEMS << ">;\n"
    << "var<workgroup> sh_idxs: array<u32, " << LARGE_SORT_BLOCK_ELEMS
    << ">;\n\n";

  s << "fn pair_less(a_key: " << val_type << ", a_idx: u32, b_key: "
    << val_type << ", b_idx: u32) -> bool {\n"
    << "  return (a_key < b_key) || ((a_key == b_key) && (a_idx < b_idx));\n"
    << "}\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let axis_size = params.data0.x;\n"
    << "  let n_blocks = params.data0.y;\n"
    << "  let total_groups = params.data1.x;\n"
    << "  let gid = linear_workgroup_id(wg_id, nwg);\n"
    << "  if (gid >= total_groups) { return; }\n"
    << "  let row = gid / n_blocks;\n"
    << "  let block = gid - row * n_blocks;\n"
    << "  let row_start = row * axis_size;\n"
    << "  let block_start = block * BLOCK_ELEMS;\n"
    << "\n"
    << "  for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {\n"
    << "    let idx = lid.x * ELEMS_PER_THREAD + e;\n"
    << "    let col = block_start + idx;\n"
    << "    if (col < axis_size) {\n"
    << "      sh_keys[idx] = input[row_start + col];\n"
    << "      sh_idxs[idx] = col;\n"
    << "    } else {\n"
    << "      sh_keys[idx] = " << sort_sentinel(val_type) << ";\n"
    << "      sh_idxs[idx] = 4294967295u;\n"
    << "    }\n"
    << "  }\n"
    << "  workgroupBarrier();\n\n"
    << "  var k: u32 = 2u;\n"
    << "  loop {\n"
    << "    if (k > BLOCK_ELEMS) { break; }\n"
    << "    var j: u32 = k >> 1u;\n"
    << "    loop {\n"
    << "      if (j == 0u) { break; }\n"
    << "      for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {\n"
    << "        let t = lid.x * ELEMS_PER_THREAD + e;\n"
    << "        let partner = t ^ j;\n"
    << "        if (partner > t && partner < BLOCK_ELEMS) {\n"
    << "          let ascending = ((t & k) == 0u);\n"
    << "          let key_t = sh_keys[t];\n"
    << "          let key_p = sh_keys[partner];\n"
    << "          let idx_t = sh_idxs[t];\n"
    << "          let idx_p = sh_idxs[partner];\n"
    << "          let should_swap = (ascending && pair_less(key_p, idx_p, key_t, idx_t)) ||\n"
    << "                            (!ascending && pair_less(key_t, idx_t, key_p, idx_p));\n"
    << "          if (should_swap) {\n"
    << "            sh_keys[t] = key_p;\n"
    << "            sh_keys[partner] = key_t;\n"
    << "            sh_idxs[t] = idx_p;\n"
    << "            sh_idxs[partner] = idx_t;\n"
    << "          }\n"
    << "        }\n"
    << "      }\n"
    << "      workgroupBarrier();\n"
    << "      j = j >> 1u;\n"
    << "    }\n"
    << "    k = k << 1u;\n"
    << "  }\n\n"
    << "  for (var e: u32 = 0u; e < ELEMS_PER_THREAD; e = e + 1u) {\n"
    << "    let idx = lid.x * ELEMS_PER_THREAD + e;\n"
    << "    let col = block_start + idx;\n"
    << "    if (col < axis_size) {\n"
    << "      let out_pos = row_start + col;\n"
    << "      keys_out[out_pos] = sh_keys[idx];\n"
    << "      idxs_out[out_pos] = sh_idxs[idx];\n"
    << "    }\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

std::string make_large_sort_merge_kernel(
    const std::string& entry_name,
    const std::string& val_type) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";
  s << wgpu::wgsl_linear_thread_id();

  s << "struct LargeSortParams {\n"
    << "  data0: vec4<u32>,\n"
    << "  data1: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> keys_in: array<" << val_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read> idxs_in: array<u32>;\n"
    << "@group(0) @binding(2) var<storage, read_write> keys_out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(3) var<storage, read_write> idxs_out: array<u32>;\n"
    << "@group(0) @binding(4) var<uniform> params: LargeSortParams;\n\n";

  s << "fn pair_less(a_key: " << val_type << ", a_idx: u32, b_key: "
    << val_type << ", b_idx: u32) -> bool {\n"
    << "  return (a_key < b_key) || ((a_key == b_key) && (a_idx < b_idx));\n"
    << "}\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let axis_size = params.data0.x;\n"
    << "  let run_width = params.data0.w;\n"
    << "  let total = params.data1.x;\n"
    << "  let gid = linear_thread_id(wg_id, lid, nwg);\n"
    << "  if (gid >= total) { return; }\n"
    << "  let row = gid / axis_size;\n"
    << "  let out_col = gid - row * axis_size;\n"
    << "  let row_start = row * axis_size;\n"
    << "  let merge_width = run_width << 1u;\n"
    << "  let merge_base = (out_col / merge_width) * merge_width;\n"
    << "  let left_start = merge_base;\n"
    << "  let left_end = min(left_start + run_width, axis_size);\n"
    << "  let right_start = left_end;\n"
    << "  let right_end = min(merge_base + merge_width, axis_size);\n"
    << "  let left_len = left_end - left_start;\n"
    << "  let right_len = right_end - right_start;\n"
    << "  let rank = out_col - merge_base;\n"
    << "  let diag = rank + 1u;\n"
    << "\n"
    << "  var low: u32 = 0u;\n"
    << "  if (diag > right_len) { low = diag - right_len; }\n"
    << "  var high: u32 = diag;\n"
    << "  if (high > left_len) { high = left_len; }\n"
    << "\n"
    << "  loop {\n"
    << "    if (low >= high) { break; }\n"
    << "    let i = (low + high) >> 1u;\n"
    << "    let j = diag - i;\n"
    << "    if (i < left_len && j > 0u) {\n"
    << "      let l_pos = row_start + left_start + i;\n"
    << "      let r_pos = row_start + right_start + j - 1u;\n"
    << "      if (pair_less(keys_in[l_pos], idxs_in[l_pos], keys_in[r_pos], idxs_in[r_pos])) {\n"
    << "        low = i + 1u;\n"
    << "        continue;\n"
    << "      }\n"
    << "    }\n"
    << "    high = i;\n"
    << "  }\n"
    << "\n"
    << "  let i = low;\n"
    << "  let j = diag - i;\n"
    << "\n"
    << "  var src_pos: u32;\n"
    << "  if (i == 0u) {\n"
    << "    src_pos = row_start + right_start + j - 1u;\n"
    << "  } else if (j == 0u) {\n"
    << "    src_pos = row_start + left_start + i - 1u;\n"
    << "  } else {\n"
    << "    let l_pos = row_start + left_start + i - 1u;\n"
    << "    let r_pos = row_start + right_start + j - 1u;\n"
    << "    if (pair_less(keys_in[l_pos], idxs_in[l_pos], keys_in[r_pos], idxs_in[r_pos])) {\n"
    << "      src_pos = r_pos;\n"
    << "    } else {\n"
    << "      src_pos = l_pos;\n"
    << "    }\n"
    << "  }\n"
    << "  keys_out[gid] = keys_in[src_pos];\n"
    << "  idxs_out[gid] = idxs_in[src_pos];\n"
    << "}\n";

  return s.str();
}

std::string make_large_sort_output_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    bool argsort) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";
  s << wgpu::wgsl_linear_thread_id();

  s << "struct LargeSortParams {\n"
    << "  data0: vec4<u32>,\n"
    << "  data1: vec4<u32>,\n"
    << "}\n\n";

  if (argsort) {
    s << "@group(0) @binding(0) var<storage, read> idxs: array<u32>;\n"
      << "@group(0) @binding(1) var<storage, read_write> output: array<u32>;\n";
  } else {
    s << "@group(0) @binding(0) var<storage, read> keys: array<" << val_type
      << ">;\n"
      << "@group(0) @binding(1) var<storage, read_write> output: array<"
      << val_type << ">;\n";
  }
  s << "@group(0) @binding(2) var<uniform> params: LargeSortParams;\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let axis_size = params.data0.x;\n"
    << "  let total = params.data1.x;\n"
    << "  let gid = linear_thread_id(wg_id, lid, nwg);\n"
    << "  if (gid >= total) { return; }\n"
    << "  let row = gid / axis_size;\n"
    << "  let idx = gid - row * axis_size;\n"
    << "  let src = row * axis_size + idx;\n";
  if (argsort) {
    s << "  output[gid] = idxs[src];\n";
  } else {
    s << "  output[gid] = keys[src];\n";
  }
  s << "}\n";

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

uint32_t checked_u32(uint64_t value, const char* label) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[WebGPU sort] ") + label +
        " exceeds the WebGPU sort kernel's 32-bit indexing limit.");
  }
  return static_cast<uint32_t>(value);
}

void release_uniform_after_completion(
    wgpu::CommandEncoder& encoder,
    WGPUBuffer uniform_buf) {
  encoder.add_completed_handler(
      [uniform_buf]() { wgpu::device().uniform_pool().release(uniform_buf); });
}

std::pair<uint32_t, uint32_t> workgroup_grid_for_items(
    uint32_t work_items,
    const char* label) {
  uint32_t groups =
      (work_items + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  return wgpu::get_2d_grid(groups, label);
}

void keep_large_sort_temporaries(
    wgpu::CommandEncoder& encoder,
    const array& keys0,
    const array& keys1,
    const array& idxs0,
    const array& idxs1) {
  encoder.add_temporary(keys0);
  encoder.add_temporary(keys1);
  encoder.add_temporary(idxs0);
  encoder.add_temporary(idxs1);
}

void commit_large_sort_phase(
    wgpu::CommandEncoder& encoder,
    const array& keys0,
    const array& keys1,
    const array& idxs0,
    const array& idxs1) {
  keep_large_sort_temporaries(encoder, keys0, keys1, idxs0, idxs1);
  encoder.commit();
}

void dispatch_large_sort_gpu(
    const array& in,
    array& sort_out,
    uint32_t axis_size,
    uint32_t n_rows,
    const std::string& val_type,
    bool argsort,
    wgpu::CommandEncoder& encoder) {
  auto& dev = wgpu::device();
  auto& pool = dev.uniform_pool();

  uint32_t total_padded = checked_u32(
      static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(axis_size),
      "temporary element count");
  uint32_t n_blocks =
      (axis_size + LARGE_SORT_BLOCK_ELEMS - 1) / LARGE_SORT_BLOCK_ELEMS;
  uint32_t total_blocks = checked_u32(
      static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(n_blocks),
      "sort block count");

  Shape temp_shape{
      static_cast<int>(n_rows),
      static_cast<int>(axis_size),
  };
  array keys0(temp_shape, in.dtype(), nullptr, {});
  keys0.set_data(allocator::malloc(wgpu::wgpu_alloc_size(keys0)));

  array keys1(temp_shape, in.dtype(), nullptr, {});
  keys1.set_data(allocator::malloc(wgpu::wgpu_alloc_size(keys1)));

  array idxs0(temp_shape, uint32, nullptr, {});
  idxs0.set_data(allocator::malloc(wgpu::wgpu_alloc_size(idxs0)));

  array idxs1(temp_shape, uint32, nullptr, {});
  idxs1.set_data(allocator::malloc(wgpu::wgpu_alloc_size(idxs1)));

  encoder.set_input_array(in);
  encoder.set_output_array(keys0);
  encoder.set_output_array(keys1);
  encoder.set_output_array(idxs0);
  encoder.set_output_array(idxs1);

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer keys0_buf = wgpu::wgpu_buffer(keys0);
  WGPUBuffer keys1_buf = wgpu::wgpu_buffer(keys1);
  WGPUBuffer idxs0_buf = wgpu::wgpu_buffer(idxs0);
  WGPUBuffer idxs1_buf = wgpu::wgpu_buffer(idxs1);
  uint64_t in_buf_size = wgpu::wgpu_bind_size(in);
  uint64_t keys0_buf_size = wgpu::wgpu_bind_size(keys0);
  uint64_t keys1_buf_size = wgpu::wgpu_bind_size(keys1);
  uint64_t idxs0_buf_size = wgpu::wgpu_bind_size(idxs0);
  uint64_t idxs1_buf_size = wgpu::wgpu_bind_size(idxs1);

  std::string op_name = argsort ? "large_argsort" : "large_sort";

  // Sort each 2048-element tile with a single workgroup. This keeps the
  // workgroup memory within WebGPU's 16KiB minimum and avoids the thousands of
  // global compare-exchange dispatches from a full-row bitonic network.
  {
    std::string entry_name = op_name + "_block_" + val_type;
    WGPUShaderModule shader =
        dev.get_or_create_shader_module(entry_name, [&]() {
          return make_large_sort_block_kernel(entry_name, val_type);
        });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    LargeSortParams params{};
    params.data0[0] = axis_size;
    params.data0[1] = n_blocks;
    params.data0[2] = n_rows;
    params.data1[0] = total_blocks;

    WGPUBuffer uniform_buf =
        pool.acquire(dev.gpu_queue(), &params, sizeof(LargeSortParams));

    std::vector<std::pair<WGPUBuffer, uint64_t>> buffers{
        {in_buf, in_buf_size},
        {keys0_buf, keys0_buf_size},
        {idxs0_buf, idxs0_buf_size},
        {uniform_buf, sizeof(LargeSortParams)},
    };

    WGPUBindGroup bg =
        wgpu::create_bind_group(pe.layout, buffers, entry_name.c_str());
    auto [wg_x, wg_y] =
        wgpu::get_2d_grid(total_blocks, "[WebGPU large sort block]");
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);
    wgpuBindGroupRelease(bg);
    release_uniform_after_completion(encoder, uniform_buf);
    commit_large_sort_phase(encoder, keys0, keys1, idxs0, idxs1);
  }

  bool ping = false;
  for (uint32_t run_width = LARGE_SORT_BLOCK_ELEMS; run_width < axis_size;
       run_width <<= 1) {
    std::string entry_name = op_name + "_merge_" + val_type;
    WGPUShaderModule shader =
        dev.get_or_create_shader_module(entry_name, [&]() {
          return make_large_sort_merge_kernel(entry_name, val_type);
        });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    LargeSortParams params{};
    params.data0[0] = axis_size;
    params.data0[1] = n_blocks;
    params.data0[2] = n_rows;
    params.data0[3] = run_width;
    params.data1[0] = total_padded;

    WGPUBuffer uniform_buf =
        pool.acquire(dev.gpu_queue(), &params, sizeof(LargeSortParams));

    WGPUBuffer keys_in_buf = ping ? keys1_buf : keys0_buf;
    WGPUBuffer idxs_in_buf = ping ? idxs1_buf : idxs0_buf;
    WGPUBuffer keys_out_buf = ping ? keys0_buf : keys1_buf;
    WGPUBuffer idxs_out_buf = ping ? idxs0_buf : idxs1_buf;
    uint64_t keys_in_size = ping ? keys1_buf_size : keys0_buf_size;
    uint64_t idxs_in_size = ping ? idxs1_buf_size : idxs0_buf_size;
    uint64_t keys_out_size = ping ? keys0_buf_size : keys1_buf_size;
    uint64_t idxs_out_size = ping ? idxs0_buf_size : idxs1_buf_size;

    WGPUBindGroup bg = wgpu::create_bind_group(
        pe.layout,
        {{keys_in_buf, keys_in_size},
         {idxs_in_buf, idxs_in_size},
         {keys_out_buf, keys_out_size},
         {idxs_out_buf, idxs_out_size},
         {uniform_buf, sizeof(LargeSortParams)}},
        entry_name.c_str());
    auto [wg_x, wg_y] =
        workgroup_grid_for_items(total_padded, "[WebGPU large sort merge]");
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);
    wgpuBindGroupRelease(bg);
    release_uniform_after_completion(encoder, uniform_buf);
    commit_large_sort_phase(encoder, keys0, keys1, idxs0, idxs1);
    ping = !ping;
  }

  encoder.set_output_array(sort_out);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(sort_out);
  uint64_t out_buf_size = wgpu::wgpu_bind_size(sort_out);
  WGPUBuffer final_keys_buf = ping ? keys1_buf : keys0_buf;
  WGPUBuffer final_idxs_buf = ping ? idxs1_buf : idxs0_buf;
  uint64_t final_keys_size = ping ? keys1_buf_size : keys0_buf_size;
  uint64_t final_idxs_size = ping ? idxs1_buf_size : idxs0_buf_size;

  {
    std::string entry_name = op_name + "_output_" + val_type;
    WGPUShaderModule shader =
        dev.get_or_create_shader_module(entry_name, [&]() {
          return make_large_sort_output_kernel(entry_name, val_type, argsort);
        });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    LargeSortParams params{};
    params.data0[0] = axis_size;
    params.data0[1] = n_blocks;
    params.data0[2] = n_rows;
    params.data1[0] = total_padded;

    WGPUBuffer uniform_buf =
        pool.acquire(dev.gpu_queue(), &params, sizeof(LargeSortParams));

    std::vector<std::pair<WGPUBuffer, uint64_t>> buffers;
    if (argsort) {
      buffers.push_back({final_idxs_buf, final_idxs_size});
    } else {
      buffers.push_back({final_keys_buf, final_keys_size});
    }
    buffers.push_back({out_buf, out_buf_size});
    buffers.push_back({uniform_buf, sizeof(LargeSortParams)});

    WGPUBindGroup bg =
        wgpu::create_bind_group(pe.layout, buffers, entry_name.c_str());
    auto [wg_x, wg_y] =
        workgroup_grid_for_items(total_padded, "[WebGPU large sort output]");
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);
    wgpuBindGroupRelease(bg);
    release_uniform_after_completion(encoder, uniform_buf);
    commit_large_sort_phase(encoder, keys0, keys1, idxs0, idxs1);
  }
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

  // We need the sort axis to be contiguous (stride 1) and be the last axis.
  // If the sort axis is not the last one, we transpose so it becomes last,
  // then do a contiguous copy so the data layout is [n_rows, axis_size].
  array in = in_orig;
  auto& encoder = wgpu::get_command_encoder(s_stream);

  bool axis_is_last = (axis == ndim - 1);
  bool contiguous_last = in.flags().contiguous && in.offset() == 0 &&
      in.size() == in.data_size() && (ndim == 0 || in.strides()[ndim - 1] == 1);

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

  uint32_t n_rows_u32 = checked_u32(static_cast<uint64_t>(n_rows), "row count");
  auto& dev = wgpu::device();

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string val_type(in_wgsl);

  // Compute padded size (next power of 2)
  uint32_t n_padded = next_power_of_2(axis_size);

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

  if (axis_size <= max_axis) {
    // Build kernel
    std::string op_name = argsort ? "argsort" : "sort";
    std::string entry_name = op_name + "_" + val_type;

    WGPUShaderModule shader = dev.get_or_create_shader_module(
        entry_name,
        [&]() { return make_sort_kernel(entry_name, val_type, argsort); });
    auto pe =
        dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

    encoder.set_input_array(in);
    encoder.set_output_array(sort_out);

    // Set params
    SortParams params{};
    params.data[0] = axis_size;
    params.data[1] = n_padded;
    params.data[2] = n_rows_u32;

    auto& pool = wgpu::device().uniform_pool();
    WGPUBuffer uniform_buf =
        pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(SortParams));

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
    auto [wg_x, wg_y] = wgpu::get_2d_grid(n_rows_u32, "[WebGPU sort]");
    encoder.dispatch_compute(pe.pipeline, bg, wg_x, wg_y);

    wgpuBindGroupRelease(bg);
    release_uniform_after_completion(encoder, uniform_buf);
  } else {
    dispatch_large_sort_gpu(
        in,
        sort_out,
        axis_size,
        n_rows_u32,
        val_type,
        argsort,
        encoder);
  }

  // If we transposed, copy from temp back to out with the original layout.
  if (need_reorder) {
    // Large sorts commit internally, so keep the reordered temporary alive for
    // the copy command recorded after those phase commits.
    encoder.add_temporary(sort_out);

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
