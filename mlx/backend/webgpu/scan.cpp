// Copyright 2026 Apple Inc.
//
// WebGPU Scan (prefix sum/product/max/min) implementation.
//
// Computes inclusive or exclusive prefix operations along a given axis.
// Each thread handles one complete "row" along the scan axis sequentially.
// Rows are parallelized across threads/workgroups.
//
// For a tensor of shape [..., N, ...] scanned along axis with N elements:
//   total_rows = product of all dimensions except the scan axis
//   Each thread processes one complete row.
//
// Supports: Sum, Prod, Max, Min (forward and reverse, inclusive and exclusive).

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
// Uniform params (vec4-aligned)
// ---------------------------------------------------------------------------

struct ScanParams {
  uint32_t data[4]; // [n_rows, axis_size, stride, ndim]
  // For strided access we encode per-row index computation:
  // shape and strides of the "non-scan" dimensions so each thread
  // can reconstruct the base offset for its row.
  // We pass shape/strides of the full array plus which axis to scan.
  uint32_t data2[4]; // [shape[0..3]]
  uint32_t data3[4]; // [shape[4..7]]
  int32_t data4[4];  // [strides[0..3]]
  int32_t data5[4];  // [strides[4..7]]
  uint32_t data6[4]; // [scan_axis, 0, 0, 0]
};

// ---------------------------------------------------------------------------
// Scan op info
// ---------------------------------------------------------------------------

struct ScanOpInfo {
  const char* name;
  const char* identity_f32;
  const char* identity_i32;
  const char* identity_u32;
  // WGSL expression combining "acc" and "val"
  const char* op_expr_f32;
  const char* op_expr_i32;
  const char* op_expr_u32;
};

ScanOpInfo get_scan_op_info(Scan::ReduceType rtype) {
  switch (rtype) {
    case Scan::Sum:
      return {
          "sum",
          "0.0",
          "0",
          "0u",
          "acc + val",
          "acc + val",
          "acc + val",
      };
    case Scan::Prod:
      return {
          "prod",
          "1.0",
          "1",
          "1u",
          "acc * val",
          "acc * val",
          "acc * val",
      };
    case Scan::Max:
      return {
          "max",
          "-3.402823e+38",
          "-2147483648",
          "0u",
          "max(acc, val)",
          "max(acc, val)",
          "max(acc, val)",
      };
    case Scan::Min:
      return {
          "min",
          "3.402823e+38",
          "2147483647",
          "4294967295u",
          "min(acc, val)",
          "min(acc, val)",
          "min(acc, val)",
      };
    case Scan::LogAddExp:
      // LogAddExp: log(exp(a) + exp(b)) = max(a,b) + log1p(exp(-abs(a-b)))
      // Identity is -inf (represented as -FLT_MAX)
      return {
          "logaddexp",
          "-3.402823e+38",
          "0", // Not used for integer types
          "0u",
          // WGSL doesn't have log1p, so we use: max(acc,val) + log(1.0 + exp(-abs(acc - val)))
          // Special case: if acc == identity, result is val
          "select(max(acc, val) + log(1.0 + exp(-abs(acc - val))), val, acc <= -3.402823e+38)",
          "acc + val", // fallback, shouldn't be used
          "acc + val",
      };
  }
  // unreachable
  return {"sum", "0.0", "0", "0u", "acc + val", "acc + val", "acc + val"};
}

const char* get_identity(const ScanOpInfo& info, const std::string& wgsl_type) {
  if (wgsl_type == "f32" || wgsl_type == "f16") {
    return info.identity_f32;
  } else if (wgsl_type == "i32") {
    return info.identity_i32;
  } else {
    return info.identity_u32;
  }
}

const char* get_op_expr(
    const ScanOpInfo& info,
    const std::string& wgsl_type) {
  if (wgsl_type == "f32" || wgsl_type == "f16") {
    return info.op_expr_f32;
  } else if (wgsl_type == "i32") {
    return info.op_expr_i32;
  } else {
    return info.op_expr_u32;
  }
}

// ---------------------------------------------------------------------------
// WGSL kernel generation
// ---------------------------------------------------------------------------

// Generate a scan kernel for contiguous data (stride along scan axis == 1).
// The data is laid out as n_rows consecutive segments of axis_size elements.
std::string make_contiguous_scan_kernel(
    const std::string& entry_name,
    const std::string& type,
    const ScanOpInfo& op_info,
    bool reverse,
    bool inclusive) {
  std::ostringstream s;

  if (type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct ScanParams {\n"
    << "  data: vec4<u32>,\n"   // [n_rows, axis_size, stride, ndim]
    << "  data2: vec4<u32>,\n"  // shape[0..3]
    << "  data3: vec4<u32>,\n"  // shape[4..7]
    << "  data4: vec4<i32>,\n"  // strides[0..3]
    << "  data5: vec4<i32>,\n"  // strides[4..7]
    << "  data6: vec4<u32>,\n"  // [scan_axis, 0, 0, 0]
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<" << type
    << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: ScanParams;\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let row = gid.x;\n"
    << "  let n_rows = params.data.x;\n"
    << "  let axis_size = params.data.y;\n"
    << "  if (row >= n_rows) { return; }\n"
    << "\n"
    << "  let row_start = row * axis_size;\n"
    << "  var acc: " << type << " = " << type << "("
    << get_identity(op_info, type) << ");\n"
    << "\n";

  // The scan loop
  if (reverse) {
    s << "  for (var i: u32 = 0u; i < axis_size; i = i + 1u) {\n"
      << "    let idx = row_start + (axis_size - 1u - i);\n";
  } else {
    s << "  for (var i: u32 = 0u; i < axis_size; i = i + 1u) {\n"
      << "    let idx = row_start + i;\n";
  }

  s << "    let val: " << type << " = input[idx];\n";

  if (inclusive) {
    // inclusive: apply op first, then write
    s << "    acc = " << type << "("
      << get_op_expr(op_info, type) << ");\n"
      << "    output[idx] = acc;\n";
  } else {
    // exclusive: write first, then apply op
    s << "    output[idx] = acc;\n"
      << "    acc = " << type << "("
      << get_op_expr(op_info, type) << ");\n";
  }

  s << "  }\n"
    << "}\n";

  return s.str();
}

// Generate a scan kernel for strided (non-contiguous) data.
// Each thread handles one row. The row's elements are located via
// a base offset + i * axis_stride, where the base is computed from
// the row index using the shape/strides of all non-scan dimensions.
std::string make_strided_scan_kernel(
    const std::string& entry_name,
    const std::string& type,
    const ScanOpInfo& op_info,
    bool reverse,
    bool inclusive) {
  std::ostringstream s;

  if (type == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << "struct ScanParams {\n"
    << "  data: vec4<u32>,\n"   // [n_rows, axis_size, axis_stride, ndim]
    << "  data2: vec4<u32>,\n"  // shape[0..3]  (non-scan dims)
    << "  data3: vec4<u32>,\n"  // shape[4..7]
    << "  data4: vec4<i32>,\n"  // strides[0..3]
    << "  data5: vec4<i32>,\n"  // strides[4..7]
    << "  data6: vec4<u32>,\n"  // [scan_axis, 0, 0, 0]
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> input: array<" << type
    << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<" << type
    << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: ScanParams;\n\n";

  // Helper to get shape/stride from uniform params
  s << "fn get_shape(d: u32) -> u32 {\n"
    << "  if (d < 4u) {\n"
    << "    return params.data2[d];\n"
    << "  } else {\n"
    << "    return params.data3[d - 4u];\n"
    << "  }\n"
    << "}\n\n";

  s << "fn get_stride(d: u32) -> i32 {\n"
    << "  if (d < 4u) {\n"
    << "    return params.data4[d];\n"
    << "  } else {\n"
    << "    return params.data5[d - 4u];\n"
    << "  }\n"
    << "}\n\n";

  // Compute the base offset for a given row index, skipping the scan axis
  s << "fn row_offset(row_idx: u32, ndim: u32, scan_axis: u32) -> u32 {\n"
    << "  var offset: u32 = 0u;\n"
    << "  var idx = row_idx;\n"
    << "  // Iterate dims in reverse, skipping scan_axis\n"
    << "  for (var d_rev: u32 = 0u; d_rev < ndim; d_rev = d_rev + 1u) {\n"
    << "    let d = ndim - 1u - d_rev;\n"
    << "    if (d == scan_axis) { continue; }\n"
    << "    let dim_size = get_shape(d);\n"
    << "    let dim_stride = get_stride(d);\n"
    << "    let coord = idx % dim_size;\n"
    << "    idx = idx / dim_size;\n"
    << "    offset = offset + u32(i32(coord) * dim_stride);\n"
    << "  }\n"
    << "  return offset;\n"
    << "}\n\n";

  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let row = gid.x;\n"
    << "  let n_rows = params.data.x;\n"
    << "  let axis_size = params.data.y;\n"
    << "  let axis_stride = params.data.z;\n"
    << "  let ndim = params.data.w;\n"
    << "  let scan_axis = params.data6.x;\n"
    << "  if (row >= n_rows) { return; }\n"
    << "\n"
    << "  let base = row_offset(row, ndim, scan_axis);\n"
    << "  var acc: " << type << " = " << type << "("
    << get_identity(op_info, type) << ");\n"
    << "\n";

  if (reverse) {
    s << "  for (var i: u32 = 0u; i < axis_size; i = i + 1u) {\n"
      << "    let idx = base + (axis_size - 1u - i) * axis_stride;\n";
  } else {
    s << "  for (var i: u32 = 0u; i < axis_size; i = i + 1u) {\n"
      << "    let idx = base + i * axis_stride;\n";
  }

  s << "    let val: " << type << " = input[idx];\n";

  if (inclusive) {
    s << "    acc = " << type << "("
      << get_op_expr(op_info, type) << ");\n"
      << "    output[idx] = acc;\n";
  } else {
    s << "    output[idx] = acc;\n"
      << "    acc = " << type << "("
      << get_op_expr(op_info, type) << ");\n";
  }

  s << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Scan::eval_gpu
// ---------------------------------------------------------------------------

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& s = stream();

  auto in = inputs[0];

  // Check if the scan axis is contiguous (stride == 1)
  bool contiguous = in.flags().contiguous && in.strides()[axis_] == 1;
  bool strided_ok =
      in.flags().contiguous && in.strides()[axis_] != 0;

  if (!contiguous && !strided_ok) {
    // Need a contiguous copy
    in = contiguous_copy_gpu(in, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(in);
    // After copy, the data is row-contiguous with standard strides
    contiguous = (in.strides()[axis_] == 1);
    strided_ok = true;
  }

  // Allocate output with same layout as input
  // WebGPU doesn't allow aliasing input/output buffers, so always allocate fresh
  out.set_data(
      allocator::malloc(wgpu::wgpu_alloc_size(in)),
      in.data_size(),
      in.strides(),
      in.flags());

  int axis_size = in.shape(axis_);
  if (axis_size == 0 || in.size() == 0) {
    return;
  }

  // Compute n_rows = total elements / axis_size
  size_t n_rows = 1;
  for (int d = 0; d < in.ndim(); d++) {
    if (d != axis_) {
      n_rows *= in.shape(d);
    }
  }

  if (n_rows == 0) {
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  const char* wgsl_type_cstr = wgpu::dtype_to_wgsl_safe(in.dtype());
  std::string type(wgsl_type_cstr);

  auto op_info = get_scan_op_info(reduce_type_);

  std::string prefix = contiguous ? "contig_scan_" : "strided_scan_";
  std::string entry_name = prefix + op_info.name + "_"
      + (reverse_ ? "rev_" : "fwd_")
      + (inclusive_ ? "inc_" : "exc_")
      + type;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() {
        if (contiguous) {
          return make_contiguous_scan_kernel(
              entry_name, type, op_info, reverse_, inclusive_);
        } else {
          return make_strided_scan_kernel(
              entry_name, type, op_info, reverse_, inclusive_);
        }
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Fill params
  ScanParams params{};
  params.data[0] = static_cast<uint32_t>(n_rows);
  params.data[1] = static_cast<uint32_t>(axis_size);
  params.data[2] = static_cast<uint32_t>(in.strides()[axis_]);
  params.data[3] = static_cast<uint32_t>(in.ndim());

  // Fill shape and strides for all dims (used by strided kernel)
  for (int d = 0; d < in.ndim() && d < static_cast<int>(wgpu::MAX_NDIM); d++) {
    if (d < 4) {
      params.data2[d] = static_cast<uint32_t>(in.shape(d));
      params.data4[d] = static_cast<int32_t>(in.strides()[d]);
    } else {
      params.data3[d - 4] = static_cast<uint32_t>(in.shape(d));
      params.data5[d - 4] = static_cast<int32_t>(in.strides()[d]);
    }
  }
  params.data6[0] = static_cast<uint32_t>(axis_);

  auto& pool = wgpu::device().uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(wgpu::device().gpu_queue(), &params, sizeof(ScanParams));

  WGPUBuffer in_buf = wgpu::wgpu_buffer(in);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t in_buf_size = wgpuBufferGetSize(in_buf);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{in_buf, in_buf_size},
       {out_buf, out_buf_size},
       {uniform_buf, sizeof(ScanParams)}});

  // Dispatch: one thread per row, ceil(n_rows / WORKGROUP_SIZE) workgroups
  uint32_t n_workgroups =
      (static_cast<uint32_t>(n_rows) + wgpu::WORKGROUP_SIZE - 1) /
      wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, n_workgroups);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core
