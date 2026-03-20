// Copyright 2026 Apple Inc.
//
// WebGPU indexing operations: Gather, GatherAxis, Scatter, ScatterAxis.
//
// Uses dynamic WGSL code generation. Each unique configuration produces
// a distinct shader module and compute pipeline, cached for reuse.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <cstring>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mlx::core {

namespace {

constexpr uint32_t WORKGROUP_SIZE = 256;
constexpr uint32_t MAX_NDIM = 8;

WGPUBuffer create_uniform_buffer(const void* data, size_t byte_size) {
  auto& dev = wgpu::device();
  WGPUBufferDescriptor desc = {};
  desc.size = byte_size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;

  WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &desc);
  if (!buf) {
    throw std::runtime_error(
        "[WebGPU indexing] Failed to create uniform buffer");
  }
  void* mapped = wgpuBufferGetMappedRange(buf, 0, byte_size);
  std::memcpy(mapped, data, byte_size);
  wgpuBufferUnmap(buf);
  return buf;
}

WGPUBuffer create_storage_buffer(const void* data, size_t byte_size) {
  auto& dev = wgpu::device();
  WGPUBufferDescriptor desc = {};
  desc.size = byte_size;
  desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;

  WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &desc);
  if (!buf) {
    throw std::runtime_error(
        "[WebGPU indexing] Failed to create storage buffer");
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
        "[WebGPU indexing] Failed to create shader module: " + key);
  }

  cache[key] = mod;
  return mod;
}

// ---------------------------------------------------------------------------
// Gather kernel
// ---------------------------------------------------------------------------
// For each output element at flat index `gid`:
//  1. Decompose gid into (idx_part, slice_part) where
//     idx_part = gid / slice_size, slice_part = gid % slice_size
//  2. Start with the source index = 0
//  3. For each index array (axis), read the index value and add
//     index_val * src_stride[axis] to the source offset
//  4. For the slice dimensions, decompose slice_part into coordinates
//     and add coord * src_stride[dim] to the source offset
//  5. Read src[source_offset] and write to out[gid]

struct GatherParams {
  uint32_t out_size;
  uint32_t slice_size;
  uint32_t src_ndim;
  uint32_t nidx;
  uint32_t idx_ndim;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
  // Followed by variable-length data in storage buffers
};

std::string make_gather_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    const std::string& idx_type,
    int nidx,
    int src_ndim,
    int idx_ndim) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "struct GatherParams {\n"
    << "  out_size: u32,\n"
    << "  slice_size: u32,\n"
    << "  src_ndim: u32,\n"
    << "  nidx: u32,\n"
    << "  idx_ndim: u32,\n"
    << "  _pad0: u32,\n"
    << "  _pad1: u32,\n"
    << "  _pad2: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> src: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: GatherParams;\n"
    << "@group(0) @binding(3) var<storage, read> metadata: array<i32>;\n";

  // Index arrays: binding 4..4+nidx-1
  for (int i = 0; i < nidx; ++i) {
    s << "@group(0) @binding(" << (4 + i) << ") var<storage, read> idx_"
      << i << ": array<" << idx_type << ">;\n";
  }
  s << "\n";

  // Metadata layout in the storage buffer:
  // [0..src_ndim-1]: src_shape
  // [src_ndim..2*src_ndim-1]: src_strides
  // [2*src_ndim..2*src_ndim+src_ndim-1]: slice_sizes
  // [3*src_ndim..3*src_ndim+nidx-1]: axes
  // [3*src_ndim+nidx..]: index shapes & strides
  //   For each index i: idx_ndim shape values, then idx_ndim stride values

  s << "fn get_src_shape(i: u32) -> u32 { return u32(metadata[i]); }\n"
    << "fn get_src_stride(i: u32) -> i32 { return metadata[params.src_ndim + i]; }\n"
    << "fn get_slice_size_dim(i: u32) -> u32 { return u32(metadata[2u * params.src_ndim + i]); }\n"
    << "fn get_axis(i: u32) -> u32 { return u32(metadata[3u * params.src_ndim + i]); }\n";

  // Index element-to-loc for strided index access
  if (idx_ndim > 0 && nidx > 0) {
    s << "\nfn get_idx_shape(idx_i: u32, dim: u32) -> u32 {\n"
      << "  let base = 3u * params.src_ndim + params.nidx + idx_i * 2u * params.idx_ndim;\n"
      << "  return u32(metadata[base + dim]);\n"
      << "}\n"
      << "fn get_idx_stride(idx_i: u32, dim: u32) -> i32 {\n"
      << "  let base = 3u * params.src_ndim + params.nidx + idx_i * 2u * params.idx_ndim + params.idx_ndim;\n"
      << "  return metadata[base + dim];\n"
      << "}\n"
      << "fn idx_elem_to_loc(flat_idx: u32, idx_i: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var rem: u32 = flat_idx;\n"
      << "  for (var d: u32 = params.idx_ndim - 1u; d < params.idx_ndim; d = d - 1u) {\n"
      << "    let dim_size = get_idx_shape(idx_i, d);\n"
      << "    let coord = rem % dim_size;\n"
      << "    loc += i32(coord) * get_idx_stride(idx_i, d);\n"
      << "    rem = rem / dim_size;\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n";
  }

  // Helper to read from the i-th index array
  for (int i = 0; i < nidx; ++i) {
    s << "fn read_idx_" << i << "(loc: u32) -> i32 {\n"
      << "  return i32(idx_" << i << "[loc]);\n"
      << "}\n";
  }
  s << "\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let out_idx = gid.x;\n"
    << "  if (out_idx >= params.out_size) { return; }\n"
    << "\n"
    << "  let idx_part = out_idx / params.slice_size;\n"
    << "  let slice_part = out_idx % params.slice_size;\n"
    << "\n"
    << "  // Compute source offset from index arrays\n"
    << "  var src_offset: i32 = 0;\n";

  // For each index array, look up the index value and compute offset
  for (int i = 0; i < nidx; ++i) {
    if (idx_ndim > 0) {
      s << "  {\n"
        << "    let idx_loc = idx_elem_to_loc(idx_part, " << i << "u);\n"
        << "    var index_val = read_idx_" << i << "(idx_loc);\n"
        << "    let ax = get_axis(" << i << "u);\n"
        << "    // Handle negative indices\n"
        << "    let dim_size = i32(get_src_shape(ax));\n"
        << "    if (index_val < 0) { index_val = index_val + dim_size; }\n"
        << "    src_offset += index_val * get_src_stride(ax);\n"
        << "  }\n";
    }
  }

  // Add slice offset
  s << "  // Compute offset from slice coordinates\n"
    << "  var slice_rem: u32 = slice_part;\n"
    << "  for (var d: u32 = params.src_ndim - 1u; d < params.src_ndim; d = d - 1u) {\n"
    << "    let ss = get_slice_size_dim(d);\n"
    << "    if (ss > 0u) {\n"
    << "      let coord = slice_rem % ss;\n"
    << "      src_offset += i32(coord) * get_src_stride(d);\n"
    << "      slice_rem = slice_rem / ss;\n"
    << "    }\n"
    << "  }\n"
    << "\n"
    << "  out[out_idx] = src[u32(src_offset)];\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// GatherAxis kernel
// ---------------------------------------------------------------------------
// Simple single-axis gather: out[i][j][k] = src[i][idx[i][j][k]][k]
// (for axis=1, for example)

struct GatherAxisParams {
  uint32_t idx_size_pre;   // product of dims before axis
  uint32_t idx_size_axis;  // size along axis
  uint32_t idx_size_post;  // product of dims after axis
  uint32_t src_axis_size;  // src.shape(axis)
  int32_t src_axis_stride; // src.strides(axis)
  int32_t idx_axis_stride; // idx.strides(axis)
  uint32_t ndim_no_axis;   // ndim - 1
  uint32_t total_size;     // total output elements
  // Followed by metadata in storage buffer:
  // shape (ndim_no_axis), src_strides (ndim_no_axis), idx_strides (ndim_no_axis)
};

std::string make_gather_axis_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    const std::string& idx_type) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "struct GatherAxisParams {\n"
    << "  idx_size_pre: u32,\n"
    << "  idx_size_axis: u32,\n"
    << "  idx_size_post: u32,\n"
    << "  src_axis_size: u32,\n"
    << "  src_axis_stride: i32,\n"
    << "  idx_axis_stride: i32,\n"
    << "  ndim_no_axis: u32,\n"
    << "  total_size: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> src: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> idx: array<"
    << idx_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read_write> out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: GatherAxisParams;\n"
    << "@group(0) @binding(4) var<storage, read> metadata: array<i32>;\n\n";

  // Metadata: shape[ndim_no_axis], src_strides[ndim_no_axis], idx_strides[ndim_no_axis]
  s << "fn get_shape_na(i: u32) -> u32 { return u32(metadata[i]); }\n"
    << "fn get_src_stride_na(i: u32) -> i32 { return metadata[params.ndim_no_axis + i]; }\n"
    << "fn get_idx_stride_na(i: u32) -> i32 { return metadata[2u * params.ndim_no_axis + i]; }\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let flat = gid.x;\n"
    << "  if (flat >= params.total_size) { return; }\n"
    << "\n"
    << "  // Decompose flat index into (pre, axis, post)\n"
    << "  let axis_post = params.idx_size_axis * params.idx_size_post;\n"
    << "  let pre = flat / axis_post;\n"
    << "  let rem = flat % axis_post;\n"
    << "  let axis_idx = rem / params.idx_size_post;\n"
    << "  let post = rem % params.idx_size_post;\n"
    << "\n"
    << "  // Compute location in src and idx using strides\n"
    << "  // For dims other than the gather axis\n"
    << "  var src_loc: i32 = 0;\n"
    << "  var idx_loc: i32 = 0;\n"
    << "  var pre_rem = pre;\n"
    << "  var post_rem = post;\n"
    << "\n"
    << "  // Process pre-axis dims\n"
    << "  let axis_u = params.ndim_no_axis - params.idx_size_post; // not used directly\n"
    << "\n"
    << "  // Simple contiguous approach: compute src_loc and idx_loc from flat coords\n"
    << "  // Decompose (pre, post) back into nd-coords using shapes with axis removed\n"
    << "  var combined = pre * params.idx_size_post + post;\n"
    << "  for (var d: u32 = params.ndim_no_axis - 1u; d < params.ndim_no_axis; d = d - 1u) {\n"
    << "    let dim_size = get_shape_na(d);\n"
    << "    let coord = combined % dim_size;\n"
    << "    src_loc += i32(coord) * get_src_stride_na(d);\n"
    << "    idx_loc += i32(coord) * get_idx_stride_na(d);\n"
    << "    combined = combined / dim_size;\n"
    << "  }\n"
    << "\n"
    << "  // Add axis contribution for idx\n"
    << "  idx_loc += i32(axis_idx) * params.idx_axis_stride;\n"
    << "\n"
    << "  // Read the index value\n"
    << "  var index_val = i32(idx[u32(idx_loc)]);\n"
    << "  // Handle negative indices\n"
    << "  if (index_val < 0) { index_val = index_val + i32(params.src_axis_size); }\n"
    << "\n"
    << "  // Add axis contribution for src using the looked-up index\n"
    << "  src_loc += index_val * params.src_axis_stride;\n"
    << "\n"
    << "  out[flat] = src[u32(src_loc)];\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Scatter kernel
// ---------------------------------------------------------------------------
// Scatter is the inverse of gather: copy src into out, then for each update
// element, write upd[idx] to out at the indexed position.

struct ScatterParams {
  uint32_t upd_size;
  uint32_t upd_post_idx_size;
  uint32_t out_ndim;
  uint32_t nidx;
  uint32_t idx_ndim;
  uint32_t upd_ndim;
  uint32_t _pad0;
  uint32_t _pad1;
};

std::string make_scatter_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    const std::string& idx_type,
    int nidx) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "struct ScatterParams {\n"
    << "  upd_size: u32,\n"
    << "  upd_post_idx_size: u32,\n"
    << "  out_ndim: u32,\n"
    << "  nidx: u32,\n"
    << "  idx_ndim: u32,\n"
    << "  upd_ndim: u32,\n"
    << "  _pad0: u32,\n"
    << "  _pad1: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> upd: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: ScatterParams;\n"
    << "@group(0) @binding(3) var<storage, read> metadata: array<i32>;\n";

  // Index arrays: binding 4..4+nidx-1
  for (int i = 0; i < nidx; ++i) {
    s << "@group(0) @binding(" << (4 + i) << ") var<storage, read> idx_"
      << i << ": array<" << idx_type << ">;\n";
  }
  s << "\n";

  // Metadata layout:
  // [0..upd_ndim-1]: upd_shape
  // [upd_ndim..2*upd_ndim-1]: upd_strides
  // [2*upd_ndim..2*upd_ndim+out_ndim-1]: out_shape
  // [2*upd_ndim+out_ndim..2*upd_ndim+2*out_ndim-1]: out_strides
  // [2*upd_ndim+2*out_ndim..2*upd_ndim+2*out_ndim+nidx-1]: axes
  // then index shapes and strides

  s << "fn get_upd_shape(i: u32) -> u32 { return u32(metadata[i]); }\n"
    << "fn get_upd_stride(i: u32) -> i32 { return metadata[params.upd_ndim + i]; }\n"
    << "fn get_out_shape(i: u32) -> u32 { return u32(metadata[2u * params.upd_ndim + i]); }\n"
    << "fn get_out_stride(i: u32) -> i32 { return metadata[2u * params.upd_ndim + params.out_ndim + i]; }\n"
    << "fn get_axis(i: u32) -> u32 { return u32(metadata[2u * params.upd_ndim + 2u * params.out_ndim + i]); }\n\n";

  // Index shapes/strides
  if (nidx > 0) {
    s << "fn idx_meta_base() -> u32 { return 2u * params.upd_ndim + 2u * params.out_ndim + params.nidx; }\n"
      << "fn get_idx_shape(idx_i: u32, dim: u32) -> u32 {\n"
      << "  return u32(metadata[idx_meta_base() + idx_i * 2u * params.idx_ndim + dim]);\n"
      << "}\n"
      << "fn get_idx_stride(idx_i: u32, dim: u32) -> i32 {\n"
      << "  return metadata[idx_meta_base() + idx_i * 2u * params.idx_ndim + params.idx_ndim + dim];\n"
      << "}\n"
      << "fn idx_elem_to_loc(flat_idx: u32, idx_i: u32) -> u32 {\n"
      << "  var loc: i32 = 0;\n"
      << "  var rem: u32 = flat_idx;\n"
      << "  for (var d: u32 = params.idx_ndim - 1u; d < params.idx_ndim; d = d - 1u) {\n"
      << "    let dim_size = get_idx_shape(idx_i, d);\n"
      << "    let coord = rem % dim_size;\n"
      << "    loc += i32(coord) * get_idx_stride(idx_i, d);\n"
      << "    rem = rem / dim_size;\n"
      << "  }\n"
      << "  return u32(loc);\n"
      << "}\n\n";
  }

  // Read from index arrays
  for (int i = 0; i < nidx; ++i) {
    s << "fn read_idx_" << i << "(loc: u32) -> i32 {\n"
      << "  return i32(idx_" << i << "[loc]);\n"
      << "}\n";
  }
  s << "\n";

  // Update element-to-loc
  s << "fn upd_elem_to_loc(flat: u32) -> u32 {\n"
    << "  var loc: i32 = 0;\n"
    << "  var rem: u32 = flat;\n"
    << "  for (var d: u32 = params.upd_ndim - 1u; d < params.upd_ndim; d = d - 1u) {\n"
    << "    let dim_size = get_upd_shape(d);\n"
    << "    let coord = rem % dim_size;\n"
    << "    loc += i32(coord) * get_upd_stride(d);\n"
    << "    rem = rem / dim_size;\n"
    << "  }\n"
    << "  return u32(loc);\n"
    << "}\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let upd_idx = gid.x;\n"
    << "  if (upd_idx >= params.upd_size) { return; }\n"
    << "\n"
    << "  // Read the update value\n"
    << "  let upd_loc = upd_elem_to_loc(upd_idx);\n"
    << "  let val = upd[upd_loc];\n"
    << "\n"
    << "  // Compute the index part and slice part of the update\n"
    << "  let idx_part = upd_idx / params.upd_post_idx_size;\n"
    << "  let slice_part = upd_idx % params.upd_post_idx_size;\n"
    << "\n"
    << "  // Compute destination offset in out\n"
    << "  var out_offset: i32 = 0;\n";

  // For each index array, compute offset into out
  for (int i = 0; i < nidx; ++i) {
    s << "  {\n"
      << "    let idx_loc = idx_elem_to_loc(idx_part, " << i << "u);\n"
      << "    var index_val = read_idx_" << i << "(idx_loc);\n"
      << "    let ax = get_axis(" << i << "u);\n"
      << "    let dim_size = i32(get_out_shape(ax));\n"
      << "    if (index_val < 0) { index_val = index_val + dim_size; }\n"
      << "    out_offset += index_val * get_out_stride(ax);\n"
      << "  }\n";
  }

  // Add slice part offset using output strides
  s << "  // Add offset from slice coordinates\n"
    << "  var slice_rem: u32 = slice_part;\n"
    << "  for (var d: u32 = params.out_ndim - 1u; d < params.out_ndim; d = d - 1u) {\n"
    << "    let dim_size = get_out_shape(d);\n"
    << "    if (dim_size > 0u) {\n"
    << "      let coord = slice_rem % dim_size;\n"
    << "      out_offset += i32(coord) * get_out_stride(d);\n"
    << "      slice_rem = slice_rem / dim_size;\n"
    << "    }\n"
    << "  }\n"
    << "\n"
    << "  // Write (last-write-wins for conflicts)\n"
    << "  out[u32(out_offset)] = val;\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// ScatterAxis kernel
// ---------------------------------------------------------------------------

struct ScatterAxisParams {
  uint32_t idx_size_pre;
  uint32_t idx_size_axis;
  uint32_t idx_size_post;
  uint32_t out_axis_size;
  int32_t upd_axis_stride;
  int32_t idx_axis_stride;
  uint32_t ndim_no_axis;
  uint32_t total_size;
};

std::string make_scatter_axis_kernel(
    const std::string& entry_name,
    const std::string& val_type,
    const std::string& idx_type) {
  std::ostringstream s;

  if (val_type == "f16") {
    s << "enable f16;\n\n";
  }

  s << "struct ScatterAxisParams {\n"
    << "  idx_size_pre: u32,\n"
    << "  idx_size_axis: u32,\n"
    << "  idx_size_post: u32,\n"
    << "  out_axis_size: u32,\n"
    << "  upd_axis_stride: i32,\n"
    << "  idx_axis_stride: i32,\n"
    << "  ndim_no_axis: u32,\n"
    << "  total_size: u32,\n"
    << "}\n\n";

  s << "@group(0) @binding(0) var<storage, read> upd: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read> idx: array<"
    << idx_type << ">;\n"
    << "@group(0) @binding(2) var<storage, read_write> out: array<"
    << val_type << ">;\n"
    << "@group(0) @binding(3) var<uniform> params: ScatterAxisParams;\n"
    << "@group(0) @binding(4) var<storage, read> metadata: array<i32>;\n\n";

  // Metadata: shape_no_axis[ndim_no_axis], upd_strides_no_axis[ndim_no_axis],
  //           idx_strides_no_axis[ndim_no_axis], out_strides_no_axis[ndim_no_axis]
  s << "fn get_shape_na(i: u32) -> u32 { return u32(metadata[i]); }\n"
    << "fn get_upd_stride_na(i: u32) -> i32 { return metadata[params.ndim_no_axis + i]; }\n"
    << "fn get_idx_stride_na(i: u32) -> i32 { return metadata[2u * params.ndim_no_axis + i]; }\n"
    << "fn get_out_stride_na(i: u32) -> i32 { return metadata[3u * params.ndim_no_axis + i]; }\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let flat = gid.x;\n"
    << "  if (flat >= params.total_size) { return; }\n"
    << "\n"
    << "  // Decompose flat index into (pre, axis, post)\n"
    << "  let axis_post = params.idx_size_axis * params.idx_size_post;\n"
    << "  let pre = flat / axis_post;\n"
    << "  let rem = flat % axis_post;\n"
    << "  let axis_idx = rem / params.idx_size_post;\n"
    << "  let post = rem % params.idx_size_post;\n"
    << "\n"
    << "  // Compute upd, idx, and out locations for non-axis dims\n"
    << "  var upd_loc: i32 = 0;\n"
    << "  var idx_loc: i32 = 0;\n"
    << "  var out_loc: i32 = 0;\n"
    << "  var combined = pre * params.idx_size_post + post;\n"
    << "  for (var d: u32 = params.ndim_no_axis - 1u; d < params.ndim_no_axis; d = d - 1u) {\n"
    << "    let dim_size = get_shape_na(d);\n"
    << "    let coord = combined % dim_size;\n"
    << "    upd_loc += i32(coord) * get_upd_stride_na(d);\n"
    << "    idx_loc += i32(coord) * get_idx_stride_na(d);\n"
    << "    out_loc += i32(coord) * get_out_stride_na(d);\n"
    << "    combined = combined / dim_size;\n"
    << "  }\n"
    << "\n"
    << "  // Add axis contribution for upd and idx\n"
    << "  upd_loc += i32(axis_idx) * params.upd_axis_stride;\n"
    << "  idx_loc += i32(axis_idx) * params.idx_axis_stride;\n"
    << "\n"
    << "  // Read the index value\n"
    << "  var index_val = i32(idx[u32(idx_loc)]);\n"
    << "  if (index_val < 0) { index_val = index_val + i32(params.out_axis_size); }\n"
    << "\n"
    << "  // Compute output axis stride (we pass out strides with axis removed,\n"
    << "  // but need the axis stride). We store it in out_strides_no_axis at the end.\n"
    << "  // Actually, we need the out stride along the scatter axis.\n"
    << "  // For row-contiguous output, this is idx_size_post.\n"
    << "  // We compute it: out_axis_stride = get_out_stride_na for a special index.\n"
    << "  // Let's just use a simple formula for row-contiguous out.\n"
    << "  let out_axis_stride = i32(params.idx_size_post);\n"
    << "  out_loc += index_val * out_axis_stride;\n"
    << "\n"
    << "  // Write the update value\n"
    << "  out[u32(out_loc)] = upd[u32(upd_loc)];\n"
    << "}\n";

  return s.str();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build metadata buffer for Gather
std::vector<int32_t> build_gather_metadata(
    const array& src,
    const std::vector<array>& inputs,
    const std::vector<int>& axes,
    const Shape& slice_sizes,
    int nidx,
    int idx_ndim) {
  std::vector<int32_t> meta;

  // src_shape
  for (int i = 0; i < src.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(src.shape(i)));
  }
  // src_strides
  for (int i = 0; i < src.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(src.strides()[i]));
  }
  // slice_sizes
  for (int i = 0; i < src.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(
        i < static_cast<int>(slice_sizes.size()) ? slice_sizes[i] : 1));
  }
  // axes
  for (int i = 0; i < nidx; ++i) {
    meta.push_back(static_cast<int32_t>(axes[i]));
  }
  // index shapes and strides for each index array
  for (int i = 0; i < nidx; ++i) {
    const auto& idx_arr = inputs[i + 1];
    for (int d = 0; d < idx_ndim; ++d) {
      meta.push_back(static_cast<int32_t>(idx_arr.shape(d)));
    }
    for (int d = 0; d < idx_ndim; ++d) {
      meta.push_back(static_cast<int32_t>(idx_arr.strides()[d]));
    }
  }

  return meta;
}

// Build metadata buffer for Scatter
std::vector<int32_t> build_scatter_metadata(
    const array& upd,
    const array& out,
    const std::vector<array>& inputs,
    const std::vector<int>& axes,
    int nidx,
    int idx_ndim) {
  std::vector<int32_t> meta;

  // upd_shape
  for (int i = 0; i < upd.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(upd.shape(i)));
  }
  // upd_strides
  for (int i = 0; i < upd.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(upd.strides()[i]));
  }
  // out_shape
  for (int i = 0; i < out.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(out.shape(i)));
  }
  // out_strides
  for (int i = 0; i < out.ndim(); ++i) {
    meta.push_back(static_cast<int32_t>(out.strides()[i]));
  }
  // axes
  for (int i = 0; i < nidx; ++i) {
    meta.push_back(static_cast<int32_t>(axes[i]));
  }
  // index shapes and strides
  for (int i = 0; i < nidx; ++i) {
    const auto& idx_arr = inputs[i + 1];
    for (int d = 0; d < idx_ndim; ++d) {
      meta.push_back(static_cast<int32_t>(idx_arr.shape(d)));
    }
    for (int d = 0; d < idx_ndim; ++d) {
      meta.push_back(static_cast<int32_t>(idx_arr.strides()[d]));
    }
  }

  return meta;
}

// Create a bind group with dynamic number of entries
WGPUBindGroup create_bind_group(
    WGPUComputePipeline pipeline,
    const std::vector<std::pair<WGPUBuffer, uint64_t>>& buffers) {
  auto& dev = wgpu::device();
  WGPUBindGroupLayout layout =
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

  std::vector<WGPUBindGroupEntry> entries(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    entries[i] = {};
    entries[i].binding = static_cast<uint32_t>(i);
    entries[i].buffer = buffers[i].first;
    entries[i].offset = 0;
    entries[i].size = buffers[i].second;
  }

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = layout;
  bg_desc.entryCount = static_cast<uint32_t>(entries.size());
  bg_desc.entries = entries.data();

  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev.gpu_device(), &bg_desc);
  wgpuBindGroupLayoutRelease(layout);

  if (!bg) {
    throw std::runtime_error(
        "[WebGPU indexing] Failed to create bind group");
  }
  return bg;
}

} // namespace

// ===========================================================================
// Gather::eval_gpu
// ===========================================================================

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  assert(inputs.size() > 0);
  const auto& src = inputs[0];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  int nidx = inputs.size() - 1;
  int idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  uint32_t slice_size = std::accumulate(
      slice_sizes_.begin(),
      slice_sizes_.end(),
      1u,
      std::multiplies<uint32_t>());

  const char* val_type = wgpu::dtype_to_wgsl(out.dtype());
  const char* idx_type = nidx > 0 ? wgpu::dtype_to_wgsl(inputs[1].dtype()) : "i32";

  std::string entry_name = std::string("gather_") + val_type + "_" +
      idx_type + "_n" + std::to_string(nidx);
  std::string pipeline_key = entry_name;

  std::string source = make_gather_kernel(
      entry_name, val_type, idx_type, nidx, src.ndim(), idx_ndim);
  WGPUShaderModule shader = get_shader_module(pipeline_key, source);

  auto& dev = wgpu::device();
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);

  // Fill params
  GatherParams params{};
  params.out_size = static_cast<uint32_t>(out.size());
  params.slice_size = slice_size;
  params.src_ndim = static_cast<uint32_t>(src.ndim());
  params.nidx = static_cast<uint32_t>(nidx);
  params.idx_ndim = static_cast<uint32_t>(idx_ndim);

  WGPUBuffer uniform_buf =
      create_uniform_buffer(&params, sizeof(GatherParams));

  // Build metadata
  auto meta = build_gather_metadata(
      src, inputs, axes_, slice_sizes_, nidx, idx_ndim);
  // Ensure at least 4 bytes for the storage buffer
  if (meta.empty()) {
    meta.push_back(0);
  }
  WGPUBuffer meta_buf = create_storage_buffer(
      meta.data(), meta.size() * sizeof(int32_t));

  // Build bind group entries
  std::vector<std::pair<WGPUBuffer, uint64_t>> bind_entries;
  WGPUBuffer src_buf = wgpu::wgpu_buffer(src);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  bind_entries.push_back({src_buf, wgpuBufferGetSize(src_buf)});
  bind_entries.push_back({out_buf, wgpuBufferGetSize(out_buf)});
  bind_entries.push_back({uniform_buf, sizeof(GatherParams)});
  bind_entries.push_back(
      {meta_buf, meta.size() * sizeof(int32_t)});

  for (int i = 0; i < nidx; ++i) {
    WGPUBuffer idx_buf = wgpu::wgpu_buffer(inputs[i + 1]);
    bind_entries.push_back({idx_buf, wgpuBufferGetSize(idx_buf)});
  }

  WGPUBindGroup bg = create_bind_group(pipeline, bind_entries);

  uint32_t num_workgroups =
      (params.out_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  encoder.dispatch_compute(pipeline, {bg}, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
  wgpuBufferDestroy(meta_buf);
  wgpuBufferRelease(meta_buf);
}

// ===========================================================================
// GatherAxis::eval_gpu
// ===========================================================================

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  assert(inputs.size() == 2);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  const char* val_type = wgpu::dtype_to_wgsl(out.dtype());
  const char* idx_type = wgpu::dtype_to_wgsl(idx.dtype());

  std::string entry_name =
      std::string("gather_axis_") + val_type + "_" + idx_type;
  std::string pipeline_key = entry_name;

  std::string source = make_gather_axis_kernel(entry_name, val_type, idx_type);
  WGPUShaderModule shader = get_shader_module(pipeline_key, source);

  auto& dev = wgpu::device();
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(src);
  encoder.set_input_array(idx);
  encoder.set_output_array(out);

  // Compute pre/axis/post sizes
  uint32_t idx_size_pre = 1;
  uint32_t idx_size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    idx_size_pre *= static_cast<uint32_t>(idx.shape(i));
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    idx_size_post *= static_cast<uint32_t>(idx.shape(i));
  }
  uint32_t idx_size_axis = static_cast<uint32_t>(idx.shape(axis_));

  GatherAxisParams params{};
  params.idx_size_pre = idx_size_pre;
  params.idx_size_axis = idx_size_axis;
  params.idx_size_post = idx_size_post;
  params.src_axis_size = static_cast<uint32_t>(src.shape(axis_));
  params.src_axis_stride = static_cast<int32_t>(src.strides()[axis_]);
  params.idx_axis_stride = static_cast<int32_t>(idx.strides()[axis_]);
  params.ndim_no_axis = static_cast<uint32_t>(src.ndim() - 1);
  params.total_size = static_cast<uint32_t>(idx.size());

  WGPUBuffer uniform_buf =
      create_uniform_buffer(&params, sizeof(GatherAxisParams));

  // Build metadata: shape_no_axis, src_strides_no_axis, idx_strides_no_axis
  std::vector<int32_t> meta;
  for (int d = 0; d < src.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(idx.shape(d)));
    }
  }
  for (int d = 0; d < src.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(src.strides()[d]));
    }
  }
  for (int d = 0; d < idx.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(idx.strides()[d]));
    }
  }
  if (meta.empty()) {
    meta.push_back(0);
  }

  WGPUBuffer meta_buf = create_storage_buffer(
      meta.data(), meta.size() * sizeof(int32_t));

  WGPUBuffer src_buf = wgpu::wgpu_buffer(src);
  WGPUBuffer idx_buf = wgpu::wgpu_buffer(idx);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  std::vector<std::pair<WGPUBuffer, uint64_t>> bind_entries;
  bind_entries.push_back({src_buf, wgpuBufferGetSize(src_buf)});
  bind_entries.push_back({idx_buf, wgpuBufferGetSize(idx_buf)});
  bind_entries.push_back({out_buf, wgpuBufferGetSize(out_buf)});
  bind_entries.push_back({uniform_buf, sizeof(GatherAxisParams)});
  bind_entries.push_back(
      {meta_buf, meta.size() * sizeof(int32_t)});

  WGPUBindGroup bg = create_bind_group(pipeline, bind_entries);

  uint32_t num_workgroups =
      (params.total_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  encoder.dispatch_compute(pipeline, {bg}, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
  wgpuBufferDestroy(meta_buf);
  wgpuBufferRelease(meta_buf);
}

// ===========================================================================
// Scatter::eval_gpu
// ===========================================================================

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  assert(inputs.size() > 1);
  const auto& upd = inputs.back();

  // Copy src into out first
  CopyType copy_type;
  if (inputs[0].data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (inputs[0].flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(inputs[0], out, copy_type, s);

  // Empty update -- done
  if (upd.size() == 0) {
    return;
  }

  int nidx = axes_.size();
  int idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  int32_t upd_post_idx_size = std::accumulate(
      upd.shape().begin() + idx_ndim,
      upd.shape().end(),
      1,
      std::multiplies<int32_t>());

  const char* val_type = wgpu::dtype_to_wgsl(out.dtype());
  const char* idx_type = nidx > 0 ? wgpu::dtype_to_wgsl(inputs[1].dtype()) : "i32";

  std::string entry_name = std::string("scatter_") + val_type + "_" +
      idx_type + "_n" + std::to_string(nidx);
  std::string pipeline_key = entry_name;

  std::string source = make_scatter_kernel(
      entry_name, val_type, idx_type, nidx);
  WGPUShaderModule shader = get_shader_module(pipeline_key, source);

  auto& dev = wgpu::device();
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);

  // Fill params
  ScatterParams params{};
  params.upd_size = static_cast<uint32_t>(upd.size());
  params.upd_post_idx_size = static_cast<uint32_t>(upd_post_idx_size);
  params.out_ndim = static_cast<uint32_t>(out.ndim());
  params.nidx = static_cast<uint32_t>(nidx);
  params.idx_ndim = static_cast<uint32_t>(idx_ndim);
  params.upd_ndim = static_cast<uint32_t>(upd.ndim());

  WGPUBuffer uniform_buf =
      create_uniform_buffer(&params, sizeof(ScatterParams));

  auto meta = build_scatter_metadata(
      upd, out, inputs, axes_, nidx, idx_ndim);
  if (meta.empty()) {
    meta.push_back(0);
  }
  WGPUBuffer meta_buf = create_storage_buffer(
      meta.data(), meta.size() * sizeof(int32_t));

  // Build bind group
  std::vector<std::pair<WGPUBuffer, uint64_t>> bind_entries;
  WGPUBuffer upd_buf = wgpu::wgpu_buffer(upd);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  bind_entries.push_back({upd_buf, wgpuBufferGetSize(upd_buf)});
  bind_entries.push_back({out_buf, wgpuBufferGetSize(out_buf)});
  bind_entries.push_back({uniform_buf, sizeof(ScatterParams)});
  bind_entries.push_back(
      {meta_buf, meta.size() * sizeof(int32_t)});

  for (int i = 0; i < nidx; ++i) {
    WGPUBuffer idx_buf = wgpu::wgpu_buffer(inputs[i + 1]);
    bind_entries.push_back({idx_buf, wgpuBufferGetSize(idx_buf)});
  }

  WGPUBindGroup bg = create_bind_group(pipeline, bind_entries);

  uint32_t num_workgroups =
      (params.upd_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  encoder.dispatch_compute(pipeline, {bg}, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
  wgpuBufferDestroy(meta_buf);
  wgpuBufferRelease(meta_buf);
}

// ===========================================================================
// ScatterAxis::eval_gpu
// ===========================================================================

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  assert(inputs.size() == 3);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];
  const auto& upd = inputs[2];

  // Copy src into out first
  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out, copy_type, s);

  // Empty update -- done
  if (upd.size() == 0) {
    return;
  }

  const char* val_type = wgpu::dtype_to_wgsl(out.dtype());
  const char* idx_type = wgpu::dtype_to_wgsl(idx.dtype());

  std::string entry_name =
      std::string("scatter_axis_") + val_type + "_" + idx_type;
  std::string pipeline_key = entry_name;

  std::string source =
      make_scatter_axis_kernel(entry_name, val_type, idx_type);
  WGPUShaderModule shader = get_shader_module(pipeline_key, source);

  auto& dev = wgpu::device();
  WGPUComputePipeline pipeline =
      dev.get_or_create_pipeline(pipeline_key, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(upd);
  encoder.set_input_array(idx);
  encoder.set_output_array(out);

  // Compute sizes
  uint32_t idx_size_pre = 1;
  uint32_t idx_size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    idx_size_pre *= static_cast<uint32_t>(idx.shape(i));
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    idx_size_post *= static_cast<uint32_t>(idx.shape(i));
  }
  uint32_t idx_size_axis = static_cast<uint32_t>(idx.shape(axis_));

  ScatterAxisParams params{};
  params.idx_size_pre = idx_size_pre;
  params.idx_size_axis = idx_size_axis;
  params.idx_size_post = idx_size_post;
  params.out_axis_size = static_cast<uint32_t>(out.shape(axis_));
  params.upd_axis_stride = static_cast<int32_t>(upd.strides()[axis_]);
  params.idx_axis_stride = static_cast<int32_t>(idx.strides()[axis_]);
  params.ndim_no_axis = static_cast<uint32_t>(idx.ndim() - 1);
  params.total_size = static_cast<uint32_t>(idx.size());

  WGPUBuffer uniform_buf =
      create_uniform_buffer(&params, sizeof(ScatterAxisParams));

  // Build metadata: shape_no_axis, upd_strides_no_axis,
  //                 idx_strides_no_axis, out_strides_no_axis
  std::vector<int32_t> meta;
  for (int d = 0; d < idx.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(idx.shape(d)));
    }
  }
  for (int d = 0; d < upd.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(upd.strides()[d]));
    }
  }
  for (int d = 0; d < idx.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(idx.strides()[d]));
    }
  }
  for (int d = 0; d < out.ndim(); ++d) {
    if (d != axis_) {
      meta.push_back(static_cast<int32_t>(out.strides()[d]));
    }
  }
  if (meta.empty()) {
    meta.push_back(0);
  }

  WGPUBuffer meta_buf = create_storage_buffer(
      meta.data(), meta.size() * sizeof(int32_t));

  WGPUBuffer upd_buf = wgpu::wgpu_buffer(upd);
  WGPUBuffer idx_buf = wgpu::wgpu_buffer(idx);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  std::vector<std::pair<WGPUBuffer, uint64_t>> bind_entries;
  bind_entries.push_back({upd_buf, wgpuBufferGetSize(upd_buf)});
  bind_entries.push_back({idx_buf, wgpuBufferGetSize(idx_buf)});
  bind_entries.push_back({out_buf, wgpuBufferGetSize(out_buf)});
  bind_entries.push_back({uniform_buf, sizeof(ScatterAxisParams)});
  bind_entries.push_back(
      {meta_buf, meta.size() * sizeof(int32_t)});

  WGPUBindGroup bg = create_bind_group(pipeline, bind_entries);

  uint32_t num_workgroups =
      (params.total_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  encoder.dispatch_compute(pipeline, {bg}, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
  wgpuBufferDestroy(meta_buf);
  wgpuBufferRelease(meta_buf);
}

} // namespace mlx::core
