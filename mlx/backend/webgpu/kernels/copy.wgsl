// Copyright 2026 Apple Inc.
//
// Copy kernels for the MLX WebGPU backend.
//
// All storage buffers use array<u32> to avoid WGSL binding type conflicts.
// Type reinterpretation is transparent for 32-bit types.
//
// Uniform params use vec4 packing to ensure correct WGSL alignment.
// Each vec4<u32> holds 4 dimension values (shape or stride).
// Two vec4s cover MAX_NDIM=8 dimensions.
//
// Kernel variants:
//   copy_s  - Scalar broadcast: fill all output elements with src[0]
//   copy_v  - Vector copy: contiguous 1:1 copy
//   copy_g  - General strided copy: strided input, contiguous output
//   copy_gg - General-general copy: strided input, strided output

const WORKGROUP_SIZE: u32 = 256u;
const N_READS: u32 = 4u;

// Uniform parameters using vec4 packing for correct alignment.
// Layout (byte offsets):
//   0: size_ndim_offsets.x = size, .y = ndim, .z = in_offset, .w = out_offset
//  16: shape_0 = shape[0..3] (vec4<u32>, 16 bytes)
//  32: shape_1 = shape[4..7] (vec4<u32>, 16 bytes)
//  48: in_strides_0 = in_strides[0..3] (vec4<i32>, 16 bytes)
//  64: in_strides_1 = in_strides[4..7] (vec4<i32>, 16 bytes)
//  80: out_strides_0 = out_strides[0..3] (vec4<i32>, 16 bytes)
//  96: out_strides_1 = out_strides[4..7] (vec4<i32>, 16 bytes)
// Total: 112 bytes
struct CopyParams {
  size_ndim_offsets: vec4<u32>, // .x=size, .y=ndim, .z=in_offset, .w=out_offset
  shape_0: vec4<u32>,          // shape[0..3]
  shape_1: vec4<u32>,          // shape[4..7]
  in_strides_0: vec4<i32>,     // in_strides[0..3]
  in_strides_1: vec4<i32>,     // in_strides[4..7]
  out_strides_0: vec4<i32>,    // out_strides[0..3]
  out_strides_1: vec4<i32>,    // out_strides[4..7]
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: CopyParams;

// Helper accessors for the packed shape/stride arrays.
fn get_shape(i: u32) -> u32 {
  if (i < 4u) {
    return params.shape_0[i];
  }
  return params.shape_1[i - 4u];
}

fn get_in_stride(i: u32) -> i32 {
  if (i < 4u) {
    return params.in_strides_0[i];
  }
  return params.in_strides_1[i - 4u];
}

fn get_out_stride(i: u32) -> i32 {
  if (i < 4u) {
    return params.out_strides_0[i];
  }
  return params.out_strides_1[i - 4u];
}

// Type conversion modes (packed in upper 16 bits of size_ndim_offsets.y):
// 0 = raw copy (no conversion)
// 1 = bool→float (u32 0/1 → f32 0.0/1.0 bit pattern)
// 2 = float/int→bool (nonzero → u32(1), zero → u32(0))
// 3 = int→float (i32 reinterpret → f32 convert)
// 4 = float→int (f32 reinterpret → i32 convert)
// 5 = uint→float (u32 → f32 convert)
fn apply_conversion(val: u32, mode: u32) -> u32 {
  if (mode == 0u) { return val; }
  if (mode == 1u) { return select(0u, 0x3F800000u, val != 0u); }
  if (mode == 2u) { return select(0u, 1u, val != 0u); }
  if (mode == 3u) { return bitcast<u32>(f32(bitcast<i32>(val))); }
  if (mode == 4u) { return bitcast<u32>(i32(bitcast<f32>(val))); }
  if (mode == 5u) { return bitcast<u32>(f32(val)); }
  return val;
}

// ============================================================================
// elem_to_loc: maps a flat element index to a buffer offset using
// shape and strides. Walks dimensions innermost to outermost.
// ============================================================================
fn elem_to_loc_in(idx: u32, ndim: u32) -> u32 {
  var loc: i32 = 0;
  var idx_rem: u32 = idx;
  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {
    let dim_idx = idx_rem % get_shape(i);
    loc += i32(dim_idx) * get_in_stride(i);
    idx_rem = idx_rem / get_shape(i);
  }
  return u32(loc);
}

fn elem_to_loc_out(idx: u32, ndim: u32) -> u32 {
  var loc: i32 = 0;
  var idx_rem: u32 = idx;
  for (var i: u32 = ndim - 1u; i < ndim; i = i - 1u) {
    let dim_idx = idx_rem % get_shape(i);
    loc += i32(dim_idx) * get_out_stride(i);
    idx_rem = idx_rem / get_shape(i);
  }
  return u32(loc);
}

// ============================================================================
// copy_s: Scalar broadcast - fill all output elements with src[in_offset].
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_s(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid.x * N_READS;
  let size = params.size_ndim_offsets.x;
  if (base >= size) { return; }
  let in_off = params.size_ndim_offsets.z;
  let out_off = params.size_ndim_offsets.w;
  let mode = (params.size_ndim_offsets.y >> 16u) & 0xFFFFu;
  let val = apply_conversion(src[in_off], mode);
  let end = min(base + N_READS, size);
  for (var i = base; i < end; i = i + 1u) {
    dst[out_off + i] = val;
  }
}

// ============================================================================
// copy_v: Vector copy - contiguous 1:1 element copy with offsets.
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_v(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid.x * N_READS;
  let size = params.size_ndim_offsets.x;
  if (base >= size) { return; }
  let in_off = params.size_ndim_offsets.z;
  let out_off = params.size_ndim_offsets.w;
  let mode = (params.size_ndim_offsets.y >> 16u) & 0xFFFFu;
  let end = min(base + N_READS, size);
  for (var i = base; i < end; i = i + 1u) {
    dst[out_off + i] = apply_conversion(src[in_off + i], mode);
  }
}

// ============================================================================
// copy_g: General strided copy - strided input, contiguous output.
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_g(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid.x * N_READS;
  let size = params.size_ndim_offsets.x;
  if (base >= size) { return; }
  let in_off = params.size_ndim_offsets.z;
  let out_off = params.size_ndim_offsets.w;
  let ndim = params.size_ndim_offsets.y & 0xFFFFu;
  let mode = (params.size_ndim_offsets.y >> 16u) & 0xFFFFu;
  let end = min(base + N_READS, size);
  for (var i = base; i < end; i = i + 1u) {
    let in_idx = elem_to_loc_in(i, ndim);
    dst[out_off + i] = apply_conversion(src[in_off + in_idx], mode);
  }
}

// ============================================================================
// copy_gg: General-general copy - strided input AND strided output.
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_gg(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid.x * N_READS;
  let size = params.size_ndim_offsets.x;
  if (base >= size) { return; }
  let in_off = params.size_ndim_offsets.z;
  let out_off = params.size_ndim_offsets.w;
  let ndim = params.size_ndim_offsets.y & 0xFFFFu;
  let mode = (params.size_ndim_offsets.y >> 16u) & 0xFFFFu;
  let end = min(base + N_READS, size);
  for (var i = base; i < end; i = i + 1u) {
    let in_idx = elem_to_loc_in(i, ndim);
    let out_idx = elem_to_loc_out(i, ndim);
    dst[out_off + out_idx] = apply_conversion(src[in_off + in_idx], mode);
  }
}
