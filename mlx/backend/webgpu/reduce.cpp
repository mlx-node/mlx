// Copyright 2026 Apple Inc.
//
// WebGPU reduction operations (sum, prod, max, min, and, or).
//
// Three reduction strategies matching the CUDA backend:
//   1. all_reduce   - reduce entire array to a single scalar
//   2. row_reduce   - reduce along the fastest-moving (last) axis
//   3. col_reduce   - reduce along non-fastest-moving axes
//
// Uses dynamic WGSL code generation with pipeline caching.

#include "mlx/backend/common/reduce.h"
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
// Reduction op helpers for WGSL codegen
// ---------------------------------------------------------------------------

struct ReduceOpInfo {
  const char* short_name;
  const char* identity_f32;
  const char* identity_i32;
  const char* identity_u32;
  const char* identity_bool;
  // WGSL expression combining "a" and "b"
  const char* op_expr_f32;
  const char* op_expr_i32;
  const char* op_expr_u32;
  const char* op_expr_bool;
};

ReduceOpInfo get_reduce_op_info(Reduce::ReduceType rtype) {
  switch (rtype) {
    case Reduce::Sum:
      return {
          "sum",
          "0.0",
          "0",
          "0u",
          "0u",
          "a + b",
          "a + b",
          "a + b",
          "a + b",
      };
    case Reduce::Prod:
      return {
          "prod",
          "1.0",
          "1",
          "1u",
          "1u",
          "a * b",
          "a * b",
          "a * b",
          "a * b",
      };
    case Reduce::Max:
      return {
          "max",
          "-3.402823e+38",  // -FLT_MAX
          "-2147483648",    // INT_MIN
          "0u",
          "0u",
          "max(a, b)",
          "max(a, b)",
          "max(a, b)",
          "max(a, b)",
      };
    case Reduce::Min:
      return {
          "min",
          "3.402823e+38",  // FLT_MAX
          "2147483647",    // INT_MAX
          "4294967295u",   // UINT_MAX
          "1u",
          "min(a, b)",
          "min(a, b)",
          "min(a, b)",
          "min(a, b)",
      };
    case Reduce::And:
      return {
          "and",
          "1.0",
          "1",
          "1u",
          "1u",
          "select(0.0, 1.0, (a != 0.0) && (b != 0.0))",
          "select(0, 1, (a != 0) && (b != 0))",
          "select(0u, 1u, (a != 0u) && (b != 0u))",
          "select(0u, 1u, (a != 0u) && (b != 0u))",
      };
    case Reduce::Or:
      return {
          "or",
          "0.0",
          "0",
          "0u",
          "0u",
          "select(0.0, 1.0, (a != 0.0) || (b != 0.0))",
          "select(0, 1, (a != 0) || (b != 0))",
          "select(0u, 1u, (a != 0u) || (b != 0u))",
          "select(0u, 1u, (a != 0u) || (b != 0u))",
      };
  }
  throw std::runtime_error("[WebGPU reduce] Unknown reduce type");
}

const char* get_identity(const ReduceOpInfo& info, const std::string& type) {
  if (type == "f32" || type == "f16")
    return info.identity_f32;
  if (type == "i32")
    return info.identity_i32;
  if (type == "u32")
    return info.identity_u32;
  if (type == "bool")
    return info.identity_bool;
  return info.identity_f32;
}

const char*
get_op_expr(const ReduceOpInfo& info, const std::string& type) {
  if (type == "f32" || type == "f16")
    return info.op_expr_f32;
  if (type == "i32")
    return info.op_expr_i32;
  if (type == "u32")
    return info.op_expr_u32;
  if (type == "bool")
    return info.op_expr_bool;
  return info.op_expr_f32;
}

// Determine the accumulation type in WGSL. Float inputs accumulate in f32,
// bool inputs accumulate in u32 (for And/Or), integers stay as-is.
std::string get_acc_type(const std::string& in_type) {
  if (in_type == "f16")
    return "f32";
  if (in_type == "bool")
    return "u32";
  return in_type;
}

// Determine the output WGSL type for a given reduce op and input type.
std::string get_out_type(
    Reduce::ReduceType rtype,
    const std::string& in_type) {
  if (rtype == Reduce::And || rtype == Reduce::Or) {
    return "u32"; // bool stored as u32
  }
  // Sum/Prod on small integers accumulate to i32
  if ((rtype == Reduce::Sum || rtype == Reduce::Prod) &&
      (in_type == "i32" || in_type == "u32")) {
    return in_type;
  }
  if (in_type == "f16")
    return "f16";
  return in_type;
}

// Get the WGSL subgroup builtin for a reduce type, or empty string if none.
const char* get_subgroup_builtin(Reduce::ReduceType rtype) {
  switch (rtype) {
    case Reduce::Sum:
      return "subgroupAdd";
    case Reduce::Max:
      return "subgroupMax";
    case Reduce::Min:
      return "subgroupMin";
    default:
      return ""; // Prod, And, Or have no subgroup builtin
  }
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: all_reduce
// ---------------------------------------------------------------------------

// C++ struct matching the WGSL AllReduceParams layout (vec4-aligned).
struct AllReduceParams {
  uint32_t data[4]; // [input_size, pad, pad, pad]
};

std::string make_all_reduce_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& acc_type,
    const std::string& out_type,
    const ReduceOpInfo& info,
    int subgroup_size = 0,
    const std::string& subgroup_builtin = "") {
  const bool use_subgroups = subgroup_size > 0;
  std::ostringstream s;

  if (use_subgroups) {
    s << "enable subgroups;\n";
  }
  if (in_type == "f16" || out_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n";
  s << "const N_READS: u32 = " << wgpu::N_READS << "u;\n\n";

  s << "struct AllReduceParams {\n"
    << "  data: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n";
  s << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << out_type << ">;\n";
  s << "@group(0) @binding(2) var<uniform> params: AllReduceParams;\n\n";

  s << "var<workgroup> shared_data: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n\n";

  const char* identity = get_identity(info, acc_type);
  const char* op = get_op_expr(info, acc_type);

  s << "fn reduce_op(a: " << acc_type << ", b: " << acc_type << ") -> "
    << acc_type << " {\n"
    << "  return " << op << ";\n"
    << "}\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u,\n"
    << " @builtin(num_workgroups) nwg: vec3u) {\n"
    << "  let input_size = params.data.x;\n"
    << "  let tid = lid.x;\n"
    << "  let total_threads = nwg.x * WORKGROUP_SIZE;\n"
    << "  let global_tid = wg_id.x * WORKGROUP_SIZE + tid;\n"
    << "\n"
    << "  // Each thread accumulates multiple elements with strided access\n"
    << "  var acc: " << acc_type << " = " << identity << ";\n"
    << "  var idx: u32 = global_tid;\n"
    << "  while (idx < input_size) {\n"
    << "    acc = reduce_op(acc, " << acc_type << "(input[idx]));\n"
    << "    idx = idx + total_threads;\n"
    << "  }\n"
    << "\n";

  if (use_subgroups && !subgroup_builtin.empty()) {
    wgpu::emit_subgroup_reduction(
        s, "acc", "shared_data", subgroup_builtin, "reduce_op",
        wgpu::WORKGROUP_SIZE, static_cast<uint32_t>(subgroup_size));
  } else {
    s << "  // Store to shared memory\n"
      << "  shared_data[tid] = acc;\n"
      << "  workgroupBarrier();\n"
      << "\n"
      << "  // Tree reduction in shared memory\n";
    wgpu::emit_unrolled_reduction(s, "shared_data", "reduce_op");
  }

  s << "\n"
    << "  // Thread 0 writes the workgroup result\n"
    << "  if (tid == 0u) {\n"
    << "    output[wg_id.x] = " << out_type << "(shared_data[0]);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: row_reduce
// ---------------------------------------------------------------------------

// C++ struct matching the WGSL RowReduceParams layout.
// Supports both simple (contiguous) and general (strided) row reduce.
struct RowReduceParams {
  uint32_t row_size_num_rows[4];  // [row_size, num_rows, ndim, reduce_ndim]
  uint32_t shape_0[4];            // non-reduce shape[0..3]
  uint32_t shape_1[4];            // non-reduce shape[4..7]
  int32_t strides_0[4];           // non-reduce strides[0..3]
  int32_t strides_1[4];           // non-reduce strides[4..7]
  uint32_t reduce_shape_0[4];     // reduce shape[0..3]
  uint32_t reduce_shape_1[4];     // reduce shape[4..7]
  int32_t reduce_strides_0[4];    // reduce strides[0..3]
  int32_t reduce_strides_1[4];    // reduce strides[4..7]
  uint32_t non_row_reductions[4]; // [non_row_reductions, pad, pad, pad]
};

std::string make_row_reduce_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& acc_type,
    const std::string& out_type,
    const ReduceOpInfo& info,
    bool general,
    int subgroup_size = 0,
    const std::string& subgroup_builtin = "") {
  const bool use_subgroups = subgroup_size > 0;
  std::ostringstream s;

  if (use_subgroups) {
    s << "enable subgroups;\n";
  }
  if (in_type == "f16" || out_type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n";
  s << "const N_READS: u32 = " << wgpu::N_READS << "u;\n\n";

  s << "struct RowReduceParams {\n"
    << "  row_size_num_rows: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  strides_0: vec4<i32>,\n"
    << "  strides_1: vec4<i32>,\n"
    << "  reduce_shape_0: vec4<u32>,\n"
    << "  reduce_shape_1: vec4<u32>,\n"
    << "  reduce_strides_0: vec4<i32>,\n"
    << "  reduce_strides_1: vec4<i32>,\n"
    << "  non_row_reductions: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << out_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: RowReduceParams;\n\n";

  s << "var<workgroup> shared_data: array<" << acc_type
    << ", WORKGROUP_SIZE>;\n\n";

  const char* identity = get_identity(info, acc_type);
  const char* op = get_op_expr(info, acc_type);

  s << "fn reduce_op(a: " << acc_type << ", b: " << acc_type << ") -> "
    << acc_type << " {\n"
    << "  return " << op << ";\n"
    << "}\n\n";

  if (general) {
    // elem_to_loc for non-reduction axes
    s << "fn get_shape(i: u32) -> u32 {\n"
      << "  if (i < 4u) { return params.shape_0[i]; }\n"
      << "  return params.shape_1[i - 4u];\n"
      << "}\n\n"
      << "fn get_stride(i: u32) -> i32 {\n"
      << "  if (i < 4u) { return params.strides_0[i]; }\n"
      << "  return params.strides_1[i - 4u];\n"
      << "}\n\n"
      << "fn get_reduce_shape(i: u32) -> u32 {\n"
      << "  if (i < 4u) { return params.reduce_shape_0[i]; }\n"
      << "  return params.reduce_shape_1[i - 4u];\n"
      << "}\n\n"
      << "fn get_reduce_stride(i: u32) -> i32 {\n"
      << "  if (i < 4u) { return params.reduce_strides_0[i]; }\n"
      << "  return params.reduce_strides_1[i - 4u];\n"
      << "}\n\n"
      << "fn elem_to_loc(idx: u32, ndim: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_shape(i);\n"
      << "    loc += i32(dim_idx) * get_stride(i);\n"
      << "    idx_rem = idx_rem / get_shape(i);\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n"
      << "fn reduce_elem_to_loc(idx: u32, ndim: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var idx_rem: u32 = idx;\n"
      << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
      << "    let dim_idx = idx_rem % get_reduce_shape(i);\n"
      << "    loc += i32(dim_idx) * get_reduce_stride(i);\n"
      << "    idx_rem = idx_rem / get_reduce_shape(i);\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n";
  }

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let row_size = params.row_size_num_rows.x;\n"
    << "  let tid = lid.x;\n"
    << "  let out_idx = wg_id.x;\n"
    << "\n";

  if (general) {
    // General row reduce: loop over non-row reduction axes,
    // then reduce the row.
    s << "  let ndim = params.row_size_num_rows.z;\n"
      << "  let reduce_ndim = params.row_size_num_rows.w;\n"
      << "  let non_row_reductions = params.non_row_reductions.x;\n"
      << "  let base_offset = elem_to_loc(out_idx, ndim);\n"
      << "\n"
      << "  var acc: " << acc_type << " = " << identity << ";\n"
      << "  for (var nr: u32 = 0u; nr < non_row_reductions; nr = nr + 1u) {\n"
      << "    let reduce_offset = reduce_elem_to_loc(nr, reduce_ndim);\n"
      << "    for (var i: u32 = tid; i < row_size; i = i + WORKGROUP_SIZE) {\n"
      << "      let idx = base_offset + reduce_offset + i;\n"
      << "      acc = reduce_op(acc, " << acc_type << "(input[idx]));\n"
      << "    }\n"
      << "  }\n";
  } else {
    // Simple (contiguous) row reduce: one workgroup per row.
    s << "  let row_start = out_idx * row_size;\n"
      << "  var acc: " << acc_type << " = " << identity << ";\n"
      << "  for (var i: u32 = tid; i < row_size; i = i + WORKGROUP_SIZE) {\n"
      << "    let idx = row_start + i;\n"
      << "    if (idx < row_start + row_size) {\n"
      << "      acc = reduce_op(acc, " << acc_type << "(input[idx]));\n"
      << "    }\n"
      << "  }\n";
  }

  s << "\n";

  if (use_subgroups && !subgroup_builtin.empty()) {
    wgpu::emit_subgroup_reduction(
        s, "acc", "shared_data", subgroup_builtin, "reduce_op",
        wgpu::WORKGROUP_SIZE, static_cast<uint32_t>(subgroup_size));
  } else {
    s << "  // Store to shared memory\n"
      << "  shared_data[tid] = acc;\n"
      << "  workgroupBarrier();\n"
      << "\n"
      << "  // Tree reduction in shared memory\n";
    wgpu::emit_unrolled_reduction(s, "shared_data", "reduce_op");
  }

  s << "\n"
    << "  // Thread 0 writes the output\n"
    << "  if (tid == 0u) {\n"
    << "    output[out_idx] = " << out_type << "(shared_data[0]);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: col_reduce
// ---------------------------------------------------------------------------

// C++ struct matching the WGSL ColReduceParams layout.
struct ColReduceParams {
  uint32_t data[4];       // [reduction_size, reduction_stride, ndim, reduce_ndim]
  uint32_t shape_0[4];    // non-reduce shape[0..3]
  uint32_t shape_1[4];    // non-reduce shape[4..7]
  int32_t strides_0[4];   // non-reduce strides[0..3]
  int32_t strides_1[4];   // non-reduce strides[4..7]
  uint32_t reduce_shape_0[4];  // reduce shape[0..3]
  uint32_t reduce_shape_1[4];  // reduce shape[4..7]
  int32_t reduce_strides_0[4]; // reduce strides[0..3]
  int32_t reduce_strides_1[4]; // reduce strides[4..7]
  uint32_t extra[4];      // [non_col_reductions, out_size, pad, pad]
};

std::string make_col_reduce_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& acc_type,
    const std::string& out_type,
    const ReduceOpInfo& info) {
  std::ostringstream s;

  if (in_type == "f16" || out_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct ColReduceParams {\n"
    << "  data: vec4<u32>,\n"
    << "  shape_0: vec4<u32>,\n"
    << "  shape_1: vec4<u32>,\n"
    << "  strides_0: vec4<i32>,\n"
    << "  strides_1: vec4<i32>,\n"
    << "  reduce_shape_0: vec4<u32>,\n"
    << "  reduce_shape_1: vec4<u32>,\n"
    << "  reduce_strides_0: vec4<i32>,\n"
    << "  reduce_strides_1: vec4<i32>,\n"
    << "  extra: vec4<u32>,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << in_type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<"
    << out_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: ColReduceParams;\n\n";

  const char* identity = get_identity(info, acc_type);
  const char* op = get_op_expr(info, acc_type);

  s << "fn reduce_op(a: " << acc_type << ", b: " << acc_type << ") -> "
    << acc_type << " {\n"
    << "  return " << op << ";\n"
    << "}\n\n";

  // elem_to_loc for non-reduction output shape
  s << "fn get_shape(i: u32) -> u32 {\n"
    << "  if (i < 4u) { return params.shape_0[i]; }\n"
    << "  return params.shape_1[i - 4u];\n"
    << "}\n\n"
    << "fn get_stride(i: u32) -> i32 {\n"
    << "  if (i < 4u) { return params.strides_0[i]; }\n"
    << "  return params.strides_1[i - 4u];\n"
    << "}\n\n"
    << "fn get_reduce_shape(i: u32) -> u32 {\n"
    << "  if (i < 4u) { return params.reduce_shape_0[i]; }\n"
    << "  return params.reduce_shape_1[i - 4u];\n"
    << "}\n\n"
    << "fn get_reduce_stride(i: u32) -> i32 {\n"
    << "  if (i < 4u) { return params.reduce_strides_0[i]; }\n"
    << "  return params.reduce_strides_1[i - 4u];\n"
    << "}\n\n"
    << "fn elem_to_loc(idx: u32, ndim: u32) -> u32 {\n"
    << "  var loc: i32 = 0;\n"
    << "  var idx_rem: u32 = idx;\n"
    << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
    << "    let dim_idx = idx_rem % get_shape(i);\n"
    << "    loc += i32(dim_idx) * get_stride(i);\n"
    << "    idx_rem = idx_rem / get_shape(i);\n"
    << "  }\n"
    << "  return u32(loc);\n"
    << "}\n\n"
    << "fn reduce_elem_to_loc(idx: u32, ndim: u32) -> u32 {\n"
    << "  var loc: i32 = 0;\n"
    << "  var idx_rem: u32 = idx;\n"
    << "  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {\n"
    << "    let dim_idx = idx_rem % get_reduce_shape(i);\n"
    << "    loc += i32(dim_idx) * get_reduce_stride(i);\n"
    << "    idx_rem = idx_rem / get_reduce_shape(i);\n"
    << "  }\n"
    << "  return u32(loc);\n"
    << "}\n\n";

  // Each thread handles one output element, loops over the reduction axis.
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let reduction_size = params.data.x;\n"
    << "  let reduction_stride = params.data.y;\n"
    << "  let ndim = params.data.z;\n"
    << "  let reduce_ndim = params.data.w;\n"
    << "  let non_col_reductions = params.extra.x;\n"
    << "  let out_size = params.extra.y;\n"
    << "\n"
    << "  let out_idx = gid.x;\n"
    << "  if (out_idx >= out_size) { return; }\n"
    << "\n"
    << "  // Compute input base offset from non-reduction indices\n"
    << "  var in_offset: u32 = 0u;\n"
    << "  if (ndim > 0u) {\n"
    << "    in_offset = elem_to_loc(out_idx, ndim);\n"
    << "  }\n"
    << "\n"
    << "  var acc: " << acc_type << " = " << identity << ";\n"
    << "\n"
    << "  // Loop over all reduction elements\n"
    << "  let total_reductions = non_col_reductions * reduction_size;\n"
    << "  for (var r: u32 = 0u; r < total_reductions; r = r + 1u) {\n"
    << "    let reduce_offset = reduce_elem_to_loc(r, reduce_ndim);\n"
    << "    let idx = in_offset + reduce_offset;\n"
    << "    acc = reduce_op(acc, " << acc_type << "(input[idx]));\n"
    << "  }\n"
    << "\n"
    << "  output[out_idx] = " << out_type << "(acc);\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// WGSL kernel generation: init_reduce
// ---------------------------------------------------------------------------

std::string make_init_reduce_kernel(
    const std::string& entry_name,
    const std::string& out_type,
    const ReduceOpInfo& info) {
  std::ostringstream s;

  if (out_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct InitParams {\n"
    << "  data: vec4<u32>,\n" // [size, pad, pad, pad]
    << "}\n\n";

  // Only 2 bindings: output buffer + uniform (no input needed)
  s << "@group(0) @binding(0) var<storage, read_write> output: array<"
    << out_type << ">;\n"
    << "@group(0) @binding(1) var<uniform> params: InitParams;\n\n";

  // Determine the identity for the output type
  const char* identity = get_identity(info, out_type);

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x;\n"
    << "  let size = params.data.x;\n"
    << "  if (idx >= size) { return; }\n"
    << "  output[idx] = " << out_type << "(" << identity << ");\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Host dispatch: init_reduce (fill output with identity)
// ---------------------------------------------------------------------------

void gpu_init_reduce(
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const Stream& s) {
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  }

  if (out.size() == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  // WGSL storage buffers don't support bool; use u32 instead
  std::string in_type_str = (std::string(in_wgsl) == "bool") ? "u32" : in_wgsl;
  std::string out_type = get_out_type(reduce_type, in_type_str);
  auto info = get_reduce_op_info(reduce_type);

  std::string entry_name =
      std::string("init_reduce_") + info.short_name + "_" + out_type;
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_init_reduce_kernel(entry_name, out_type, info); });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_output_array(out);

  // Fill uniform
  auto& pool = wgpu::device().uniform_pool();
  AllReduceParams params{};
  params.data[0] = static_cast<uint32_t>(out.size());
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(AllReduceParams));

  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{out_buf, out_buf_size},
       {uniform_buf, sizeof(AllReduceParams)}});

  uint32_t num_workgroups =
      (static_cast<uint32_t>(out.size()) + wgpu::WORKGROUP_SIZE - 1) /
      wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

// ---------------------------------------------------------------------------
// Host dispatch: all_reduce
// ---------------------------------------------------------------------------

void gpu_all_reduce(
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const Stream& s) {
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  // WGSL storage buffers don't support bool; use u32 instead
  std::string in_type = (std::string(in_wgsl) == "bool") ? "u32" : in_wgsl;
  std::string acc_type = get_acc_type(in_type);
  std::string out_type = get_out_type(reduce_type, in_type);
  auto info = get_reduce_op_info(reduce_type);

  // Check subgroup support
  const char* sg_builtin = get_subgroup_builtin(reduce_type);
  int sg = (sg_builtin[0] != '\0') ? wgpu::effective_subgroup_size() : 0;
  std::string sg_suffix = (sg > 0) ? wgpu::subgroup_suffix() : "";

  // ContiguousAllReduce guarantees size() == data_size()
  uint32_t input_size = static_cast<uint32_t>(in.size());
  uint32_t elems_per_wg = wgpu::WORKGROUP_SIZE * wgpu::N_READS;
  uint32_t num_workgroups = (input_size + elems_per_wg - 1) / elems_per_wg;

  // Clamp to reasonable number of workgroups
  if (num_workgroups > 1024) {
    num_workgroups = 1024;
  }

  std::string entry_name =
      std::string("all_reduce_") + info.short_name + "_" + in_type + sg_suffix;
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_all_reduce_kernel(
            entry_name, in_type, acc_type, out_type, info, sg, sg_builtin);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);

  auto& pool = wgpu::device().uniform_pool();

  if (num_workgroups > 1) {
    // Two-pass approach: first pass reduces to intermediate array
    array intermediate({static_cast<int>(num_workgroups)}, out.dtype(), nullptr, {});
    intermediate.set_data(allocator::malloc(wgpu::wgpu_alloc_size(intermediate)));
    encoder.add_temporary(intermediate);
    encoder.set_output_array(intermediate);

    AllReduceParams params{};
    params.data[0] = input_size;
    WGPUBuffer uniform_buf =
        pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(AllReduceParams));

    WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
    WGPUBuffer inter_buf = wgpu::wgpu_buffer(intermediate);
    uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
    uint64_t inter_buf_size = wgpuBufferGetSize(inter_buf);

    WGPUBindGroup bg = wgpu::create_bind_group(
        pe.layout,
        {{in_buf, in_buf_size},
         {inter_buf, inter_buf_size},
         {uniform_buf, sizeof(AllReduceParams)}});

    encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);
    wgpuBindGroupRelease(bg);
    encoder.add_completed_handler([uniform_buf]() {
      wgpu::device().uniform_pool().release(uniform_buf);
    });

    // Second pass: reduce intermediate -> output (single workgroup)
    // Re-generate kernel for the output type (in case intermediate is different
    // from input)
    std::string entry2 =
        std::string("all_reduce_") + info.short_name + "_" + out_type + sg_suffix;
    WGPUShaderModule shader2 = dev.get_or_create_shader_module(
        entry2,
        [&]() {
          return make_all_reduce_kernel(
              entry2, out_type, acc_type, out_type, info, sg, sg_builtin);
        });
    auto pe2 =
        dev.get_or_create_pipeline(entry2, shader2, entry2.c_str());

    encoder.set_input_array(intermediate);
    encoder.set_output_array(out);

    AllReduceParams params2{};
    params2.data[0] = num_workgroups;
    WGPUBuffer uniform_buf2 =
        pool.acquire(wgpu::device().gpu_queue(), &params2, sizeof(AllReduceParams));

    WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
    uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

    WGPUBindGroup bg2 = wgpu::create_bind_group(
        pe2.layout,
        {{inter_buf, inter_buf_size},
         {out_buf, out_buf_size},
         {uniform_buf2, sizeof(AllReduceParams)}});

    encoder.dispatch_compute(pe2.pipeline, bg2, 1);
    wgpuBindGroupRelease(bg2);
    encoder.add_completed_handler([uniform_buf2]() {
      wgpu::device().uniform_pool().release(uniform_buf2);
    });
  } else {
    // Single-pass: all elements fit in one workgroup
    encoder.set_output_array(out);

    AllReduceParams params{};
    params.data[0] = input_size;
    WGPUBuffer uniform_buf =
        pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(AllReduceParams));

    WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
    WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
    uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
    uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

    WGPUBindGroup bg = wgpu::create_bind_group(
        pe.layout,
        {{in_buf, in_buf_size},
         {out_buf, out_buf_size},
         {uniform_buf, sizeof(AllReduceParams)}});

    encoder.dispatch_compute(pe.pipeline, bg, 1);
    wgpuBindGroupRelease(bg);
    encoder.add_completed_handler([uniform_buf]() {
      wgpu::device().uniform_pool().release(uniform_buf);
    });
  }
}

// ---------------------------------------------------------------------------
// Host dispatch: row_reduce
// ---------------------------------------------------------------------------

void gpu_row_reduce(
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const Stream& s) {
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  // WGSL storage buffers don't support bool; use u32 instead
  std::string in_type = (std::string(in_wgsl) == "bool") ? "u32" : in_wgsl;
  std::string acc_type = get_acc_type(in_type);
  std::string out_type = get_out_type(reduce_type, in_type);
  auto info = get_reduce_op_info(reduce_type);

  // Check subgroup support
  const char* sg_builtin = get_subgroup_builtin(reduce_type);
  int sg = (sg_builtin[0] != '\0') ? wgpu::effective_subgroup_size() : 0;
  std::string sg_suffix = (sg > 0) ? wgpu::subgroup_suffix() : "";

  assert(!plan.shape.empty());
  uint32_t row_size = static_cast<uint32_t>(plan.shape.back());

  // Check if this is a "general" row reduce (multi-axis) or simple
  bool general = plan.shape.size() > 1;

  std::string variant_str = general ? "general" : "simple";
  std::string entry_name = std::string("row_reduce_") + variant_str + "_" +
      info.short_name + "_" + in_type + sg_suffix;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_row_reduce_kernel(
            entry_name, in_type, acc_type, out_type, info, general,
            sg, sg_builtin);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Build params
  RowReduceParams params{};
  params.row_size_num_rows[0] = row_size;
  params.row_size_num_rows[1] = static_cast<uint32_t>(out.size());

  if (general) {
    // Fill non-reduce shape/strides
    auto [shape_vec, strides_vec] = shapes_without_reduction_axes(in, axes);
    auto [cshape, cstrides] =
        collapse_contiguous_dims(shape_vec, strides_vec);

    uint32_t ndim = static_cast<uint32_t>(cshape.size());
    params.row_size_num_rows[2] = ndim;

    for (uint32_t i = 0; i < ndim && i < wgpu::MAX_NDIM; ++i) {
      if (i < 4) {
        params.shape_0[i] = static_cast<uint32_t>(cshape[i]);
        params.strides_0[i] = static_cast<int32_t>(cstrides[i]);
      } else {
        params.shape_1[i - 4] = static_cast<uint32_t>(cshape[i]);
        params.strides_1[i - 4] = static_cast<int32_t>(cstrides[i]);
      }
    }

    // Fill reduce shape/strides (excluding the last reduction axis
    // which is the row being reduced)
    uint32_t reduce_ndim =
        static_cast<uint32_t>(plan.shape.size() - 1);
    params.row_size_num_rows[3] = reduce_ndim;

    uint32_t non_row_reductions = 1;
    for (uint32_t i = 0; i < reduce_ndim; i++) {
      if (i < 4) {
        params.reduce_shape_0[i] = static_cast<uint32_t>(plan.shape[i]);
        params.reduce_strides_0[i] = static_cast<int32_t>(plan.strides[i]);
      } else {
        params.reduce_shape_1[i - 4] =
            static_cast<uint32_t>(plan.shape[i]);
        params.reduce_strides_1[i - 4] =
            static_cast<int32_t>(plan.strides[i]);
      }
      non_row_reductions *= static_cast<uint32_t>(plan.shape[i]);
    }
    params.non_row_reductions[0] = non_row_reductions;
  }

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(RowReduceParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(RowReduceParams)}});

  // One workgroup per output row
  uint32_t num_workgroups = static_cast<uint32_t>(out.size());
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

// ---------------------------------------------------------------------------
// Host dispatch: col_reduce
// ---------------------------------------------------------------------------

void gpu_col_reduce(
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const Stream& s) {
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* in_wgsl = wgpu::dtype_to_wgsl_safe(in.dtype());
  // WGSL storage buffers don't support bool; use u32 instead
  std::string in_type = (std::string(in_wgsl) == "bool") ? "u32" : in_wgsl;
  std::string acc_type = get_acc_type(in_type);
  std::string out_type = get_out_type(reduce_type, in_type);
  auto info = get_reduce_op_info(reduce_type);

  std::string entry_name =
      std::string("col_reduce_") + info.short_name + "_" + in_type;
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        return make_col_reduce_kernel(
            entry_name, in_type, acc_type, out_type, info);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Build ColReduceParams
  ColReduceParams params{};

  assert(!plan.shape.empty());
  uint32_t reduction_size = static_cast<uint32_t>(plan.shape.back());
  int64_t reduction_stride = plan.strides.back();
  params.data[0] = reduction_size;
  params.data[1] = static_cast<uint32_t>(reduction_stride);

  // Non-reduction shape/strides
  auto [shape_vec, strides_vec] = shapes_without_reduction_axes(in, axes);
  // Remove trailing axes that are covered by reduction_stride
  {
    int64_t stride_back = 1;
    while (!shape_vec.empty() && stride_back < reduction_stride) {
      stride_back *= shape_vec.back();
      shape_vec.pop_back();
      strides_vec.pop_back();
    }
  }
  auto [cshape, cstrides] =
      collapse_contiguous_dims(shape_vec, strides_vec);
  uint32_t ndim = static_cast<uint32_t>(cshape.size());
  params.data[2] = ndim;

  for (uint32_t i = 0; i < ndim && i < wgpu::MAX_NDIM; ++i) {
    if (i < 4) {
      params.shape_0[i] = static_cast<uint32_t>(cshape[i]);
      params.strides_0[i] = static_cast<int32_t>(cstrides[i]);
    } else {
      params.shape_1[i - 4] = static_cast<uint32_t>(cshape[i]);
      params.strides_1[i - 4] = static_cast<int32_t>(cstrides[i]);
    }
  }

  // Reduce shape/strides
  uint32_t reduce_ndim = static_cast<uint32_t>(plan.shape.size());
  params.data[3] = reduce_ndim;

  uint32_t non_col_reductions = 1;
  for (uint32_t i = 0; i < reduce_ndim - 1; i++) {
    non_col_reductions *= static_cast<uint32_t>(plan.shape[i]);
  }

  for (uint32_t i = 0; i < reduce_ndim && i < wgpu::MAX_NDIM; ++i) {
    if (i < 4) {
      params.reduce_shape_0[i] = static_cast<uint32_t>(plan.shape[i]);
      params.reduce_strides_0[i] = static_cast<int32_t>(plan.strides[i]);
    } else {
      params.reduce_shape_1[i - 4] =
          static_cast<uint32_t>(plan.shape[i]);
      params.reduce_strides_1[i - 4] =
          static_cast<int32_t>(plan.strides[i]);
    }
  }

  params.extra[0] = non_col_reductions;
  params.extra[1] = static_cast<uint32_t>(out.size());

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(ColReduceParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(ColReduceParams)}});

  // Each thread handles one output element
  uint32_t num_workgroups =
      (static_cast<uint32_t>(out.size()) + wgpu::WORKGROUP_SIZE - 1) /
      wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point: Reduce::eval_gpu
// ---------------------------------------------------------------------------

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here.
  assert(!axes_.empty());
  assert(out.size() != in.size());

  auto& s = stream();

  if (in.size() == 0) {
    gpu_init_reduce(in, out, reduce_type_, s);
    return;
  }

  // Get reduction plan
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  bool broadcasted = false;
  for (int i = 0, j = 0; i < in.ndim() && !broadcasted; i++) {
    if (j < static_cast<int>(axes_.size()) && axes_[j] == i) {
      j++;
    } else {
      broadcasted = in.strides()[i] == 0;
    }
  }
  if (plan.type == GeneralReduce || broadcasted || !in.flags().contiguous) {
    array in_copy = contiguous_copy_gpu(in, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  if (plan.type == ContiguousAllReduce) {
    gpu_all_reduce(in, out, reduce_type_, s);
    return;
  }

  if (plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
    gpu_row_reduce(in, out, reduce_type_, axes_, plan, s);
    return;
  }

  if (plan.type == ContiguousStridedReduce ||
      plan.type == GeneralStridedReduce) {
    gpu_col_reduce(in, out, reduce_type_, axes_, plan, s);
    return;
  }

  throw std::runtime_error("[WebGPU] No plan reached in reduce.");
}

} // namespace mlx::core
