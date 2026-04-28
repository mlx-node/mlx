// Copyright 2026 Apple Inc.

#pragma once

#include <webgpu/webgpu.h>

#include <cstring>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/dtype.h"

namespace mlx::core::wgpu {

// Shared constants used across WebGPU kernel files.
constexpr uint32_t WORKGROUP_SIZE = 256;
constexpr uint32_t MAX_NDIM = 8;
constexpr uint32_t N_READS = 4;

// Return the GPU-side element size for a dtype.
// WGSL has no 8-bit or 16-bit integer storage, so:
//   bool    -> u32 (4 bytes)
//   bfloat16 -> f32 (4 bytes)
//   everything else -> same as CPU itemsize
inline size_t wgpu_itemsize(Dtype dtype) {
  switch (dtype.val()) {
    case Dtype::Val::bool_: return 4;     // bool stored as u32
    case Dtype::Val::bfloat16: return 4;  // bf16 promoted to f32
    default: return dtype.size();
  }
}

// Return the GPU buffer allocation size needed for an array.
// Uses size() (logical element count from shape) not data_size() (memory span),
// matching the semantics of nbytes() = size() * itemsize().
inline size_t wgpu_alloc_size(const array& arr) {
  return arr.size() * wgpu_itemsize(arr.dtype());
}

// Return the GPU allocation size for a bf16 array stored in packed form
// (two bf16 per u32, with odd-size arrays rounded up to the next pair).
// This intentionally bypasses wgpu_itemsize()/wgpu_alloc_size() — the packed
// path is opt-in and never touches those helpers.
inline size_t wgpu_packed_alloc_size(const array& arr) {
  return ((arr.size() + 1) / 2) * 4;
}

// Shared WGSL MatmulParams struct definition. Must match the C++ MatmulParams
// layout in matmul.cpp (96 bytes, vec4<u32> fields at 16-byte-aligned offsets).
inline std::string wgsl_matmul_params_struct() {
  return std::string(
      "struct MatmulParams {\n"
      "  M: u32, N: u32, K: u32,\n"
      "  lda: u32, ldb: u32, ldc: u32,\n"
      "  batch_size: u32,\n"
      "  batch_ndim: u32,\n"
      "  batch_shape: vec4<u32>,\n"
      "  batch_stride_a: vec4<u32>,\n"
      "  batch_stride_b: vec4<u32>,\n"
      "  batch_stride_c: u32,\n"
      "  offset_a: u32, offset_b: u32,\n"
      "  _pad: u32,\n"
      "}\n\n"
      // Multi-dim batch index decomposition (mirrors Metal elem_to_loc_broadcast).
      // Fast path for batch_ndim==1 avoids the loop + division/modulo overhead
      // that the general path pays on every thread invocation.
      "fn elem_to_loc_broadcast(elem: u32, ndim: u32, shape: vec4<u32>, "
      "a_strides: vec4<u32>, b_strides: vec4<u32>) -> vec2<u32> {\n"
      "  if (ndim == 1u) {\n"
      "    return vec2<u32>(elem * a_strides[0], elem * b_strides[0]);\n"
      "  }\n"
      "  var loc_a: u32 = 0u;\n"
      "  var loc_b: u32 = 0u;\n"
      "  var e = elem;\n"
      "  for (var i: i32 = i32(ndim) - 1; i >= 0; i = i - 1) {\n"
      "    let s = shape[i];\n"
      "    let pos = e % s;\n"
      "    e = e / s;\n"
      "    loc_a = loc_a + pos * a_strides[i];\n"
      "    loc_b = loc_b + pos * b_strides[i];\n"
      "  }\n"
      "  return vec2<u32>(loc_a, loc_b);\n"
      "}\n\n");
}

// Shared WGSL helper used by every kernel that reads packed-bf16 storage.
// Two bf16 values are stored in a u32 as `lo | (hi << 16)`; each bf16 is the
// upper 16 bits of an f32, so reconstruction is a left-shift + bitcast.
inline std::string wgsl_unpack_bf16_pair() {
  return std::string(
      "fn unpack_bf16_pair(p: u32) -> vec2<f32> {\n"
      "  let lo_bits = (p & 0x0000FFFFu) << 16u;\n"
      "  let hi_bits = p & 0xFFFF0000u;\n"
      "  return vec2<f32>(bitcast<f32>(lo_bits), bitcast<f32>(hi_bits));\n"
      "}\n\n");
}

// Ensure the output array's buffer is large enough for GPU-side element sizes.
// Call after set_*_output_data() which allocates based on CPU itemsize.
// For types where GPU elements are larger (bool, bfloat16), this reallocates.
//
// CEL-F001: when an output is donated from an input whose physical WebGPU
// buffer is already GPU-sized (e.g. a bf16 array produced by a prior GPU
// kernel, then donated to this op's output), we must NOT reallocate — the
// donation is the whole point. The caller's `out.itemsize()` is the CPU
// itemsize (2 for bf16), but the underlying WebGPUBuffer is already the
// wider 4B/elem. Short-circuit when the existing buffer is large enough to
// hold the GPU-sized payload; otherwise fall through to the reallocation.
inline void ensure_wgpu_size(array& out) {
  size_t gpu_is = wgpu_itemsize(out.dtype());
  if (gpu_is <= out.itemsize()) {
    return;
  }
  auto* wbuf = static_cast<const WebGPUBuffer*>(out.buffer().ptr());
  size_t needed = out.data_size() * gpu_is;
  if (wbuf && wbuf->size >= needed) {
    // Physical buffer is already wide enough (either a donation from a
    // prior GPU kernel or a previous ensure_wgpu_size call). Skip the
    // reallocation — this preserves donation and avoids churning the
    // buffer pool every time a fused output is emitted.
    return;
  }
  out.set_data(
      allocator::malloc(needed),
      out.data_size(),
      out.strides(),
      out.flags());
}

// WebGPU forbids the same buffer appearing as both read-only and read-write
// storage in the same compute pass (synchronization scope). This happens when
// the framework "donates" an input buffer to the output (in-place reuse).
// Metal/CUDA allow this, but WebGPU does not. Call this after
// set_binary_op_output_data / set_unary_output_data to force a fresh
// allocation when aliasing would occur.
inline void ensure_no_alias(array& out, const array& in) {
  if (out.data_shared_ptr() == in.data_shared_ptr()) {
    out.set_data(
        allocator::malloc(out.data_size() * wgpu_itemsize(out.dtype())),
        out.data_size(),
        out.strides(),
        out.flags());
  }
}

// Poll the WebGPU instance to process pending async operations.
inline void poll_instance(WGPUInstance instance) {
#if defined(__wasm__)
  // WASI_IMPORT: JS bridge handles polling; this is a no-op.
  (void)instance;
#elif defined(WEBGPU_BACKEND_WGPU)
  wgpuInstancePoll(instance, false, nullptr);
#else
  wgpuInstanceProcessEvents(instance);
#endif
}

// Create a WGPUBuffer with initial data for the given usage flags.
inline WGPUBuffer create_buffer_with_data(
    const void* data,
    size_t byte_size,
    WGPUBufferUsageFlags usage) {
  auto& dev = device();
  WGPUBufferDescriptor desc = {};
  desc.size = byte_size;
  desc.usage = usage;
  desc.mappedAtCreation = true;

  WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.gpu_device(), &desc);
  if (!buf) {
    throw std::runtime_error("[WebGPU] Failed to create buffer");
  }
  void* mapped = wgpuBufferGetMappedRange(buf, 0, byte_size);
  if (!mapped) {
    wgpuBufferRelease(buf);
    throw std::runtime_error("[WebGPU] Failed to get mapped range");
  }
  std::memcpy(mapped, data, byte_size);
  wgpuBufferUnmap(buf);
  return buf;
}

// Create a uniform buffer with initial data.
inline WGPUBuffer create_uniform_buffer(const void* data, size_t byte_size) {
  return create_buffer_with_data(
      data, byte_size, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
}

// Create a storage buffer with initial data.
inline WGPUBuffer create_storage_buffer(const void* data, size_t byte_size) {
  return create_buffer_with_data(
      data, byte_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
}

// Create a bind group with a dynamic number of buffer entries.
// Each pair is (buffer, size). Bindings are assigned sequentially from 0.
// `label` is optional diagnostic text surfaced in the error message on
// failure; callers typically pass the kernel name. Default empty string
// preserves the previous minimal error text for unlabeled call sites.
inline WGPUBindGroup create_bind_group(
    WGPUBindGroupLayout layout,
    const std::vector<std::pair<WGPUBuffer, uint64_t>>& buffers,
    const char* label = nullptr) {
  auto& dev = device();

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

  if (!bg) {
    std::string msg = "[WebGPU] Failed to create bind group";
    if (label && *label) {
      msg += " (kernel=";
      msg += label;
      msg += ", entries=";
      msg += std::to_string(buffers.size());
      msg += ")";
    }
    throw std::runtime_error(msg);
  }
  return bg;
}

// Extract the WGPUBuffer from an array's allocator buffer.
// If the buffer has dirty CPU data (written via raw_ptr() but never uploaded),
// this uploads it to the GPU via wgpuQueueWriteBuffer before returning.
WGPUBuffer wgpu_buffer(const array& arr);

// Compute the byte offset into the array's buffer.
inline uint64_t wgpu_offset(const array& arr) {
  return static_cast<uint64_t>(arr.offset());
}

// Compute the data size in bytes for an array.
inline uint64_t wgpu_data_size(const array& arr) {
  return static_cast<uint64_t>(arr.data_size()) * wgpu_itemsize(arr.dtype());
}

// Compute a bind-group range large enough for a shader that indexes from the
// start of the underlying buffer and applies arr.offset() in element units.
// Avoids a wgpuBufferGetSize round trip on the browser bridge hot path.
inline uint64_t wgpu_bind_size(const array& arr) {
  const auto* wbuf = static_cast<const WebGPUBuffer*>(arr.buffer().ptr());
  if (wbuf && wbuf->size > 0) {
    return static_cast<uint64_t>(wbuf->size);
  }

  const uint64_t elem_offset =
      static_cast<uint64_t>(arr.offset()) / static_cast<uint64_t>(arr.itemsize());
  uint64_t max_elem = elem_offset;
  for (int i = 0; i < arr.ndim(); ++i) {
    if (arr.shape(i) == 0) {
      return 0;
    }
    const auto stride = arr.strides()[i];
    if (stride > 0) {
      max_elem += static_cast<uint64_t>(arr.shape(i) - 1) *
          static_cast<uint64_t>(stride);
    }
  }
  const uint64_t elem_end = max_elem + 1;
  if (wbuf && wbuf->storage_mode == StorageMode::PackedBf16) {
    return ((elem_end + 1) / 2) * 4;
  }
  return elem_end * wgpu_itemsize(arr.dtype());
}

// Map MLX dtype to WGSL type name string.
inline const char* dtype_to_wgsl(Dtype dtype) {
  switch (dtype.val()) {
    case Dtype::Val::bool_:
      return "u32"; // Bool stored as u32 in WebGPU
    case Dtype::Val::uint8:
      throw std::runtime_error(
          "[WebGPU] uint8 is not supported (no 8-bit storage in WGSL)");
    case Dtype::Val::uint16:
      throw std::runtime_error(
          "[WebGPU] uint16 is not supported (no 16-bit integer storage in WGSL)");
    case Dtype::Val::uint32:
      return "u32";
    case Dtype::Val::uint64:
      throw std::runtime_error(
          "[WebGPU] uint64 is not supported on WebGPU backend");
    case Dtype::Val::int8:
      throw std::runtime_error(
          "[WebGPU] int8 is not supported (no 8-bit storage in WGSL)");
    case Dtype::Val::int16:
      throw std::runtime_error(
          "[WebGPU] int16 is not supported (no 16-bit integer storage in WGSL)");
    case Dtype::Val::int32:
      return "i32";
    case Dtype::Val::int64:
      throw std::runtime_error(
          "[WebGPU] int64 is not supported on WebGPU backend");
    case Dtype::Val::float16:
      return "f16";
    case Dtype::Val::bfloat16:
      return "f32"; // Interpreted as f32; caller must ensure data is promoted
    case Dtype::Val::float32:
      return "f32";
    case Dtype::Val::float64:
      throw std::runtime_error(
          "[WebGPU] float64 is not supported on WebGPU backend");
    case Dtype::Val::complex64:
      throw std::runtime_error(
          "[WebGPU] complex64 is not supported (needs custom struct in WGSL)");
    default:
      throw std::runtime_error("[wgpu] Unsupported dtype for WGSL");
  }
}

// Device-aware dtype helper: returns the WGSL type name for a dtype, falling
// back to "f32" for f16 when the device has no `shader-f16` feature.
//
// CG-F012: the old fallback just returned "f32" by name while wgpu_itemsize
// kept reporting 2 bytes for f16. Kernels emitted with array<f32> then read
// 4 bytes per element from a buffer that only holds 2-byte elements — silent
// garbage past the first half. Fully fixing this requires:
//
//   1. wgpu_itemsize(f16) returning 4 when shader-f16 is absent, AND
//   2. upload_with_conversion() in device.cpp widening f16 → f32 on upload
//      (mirroring the existing bf16 → f32 path).
//
// (2) lives in device.cpp, which is off-limits to this change cluster. Until
// that lands, we fail closed: throw at shader-generation time rather than
// silently emit a kernel whose reads will return garbage. Devices without
// `shader-f16` simply cannot execute an f16 workload today — this makes that
// explicit instead of producing incorrect numerics.
//
// TODO: coordinate with device.cpp: add an `f16 → f32` widening branch in
// `upload_with_conversion()` and flip this throw to return "f32" once the
// upload side can materialise the buffer with f32 elements.
inline const char* dtype_to_wgsl_safe(Dtype dtype) {
  if (dtype.val() == Dtype::Val::float16 && !device().has_shader_f16()) {
    throw std::runtime_error(
        "[WebGPU] float16 kernel requested on a device without the "
        "'shader-f16' feature. A widened-upload fallback is not yet "
        "implemented (see CG-F012). Re-run on a device that reports "
        "shader-f16, or cast the input to float32 before the op.");
  }
  return dtype_to_wgsl(dtype);
}

// Emit "enable f16;\n\n" if any of the given types is "f16"
inline void emit_f16_enable(std::ostringstream& s, std::initializer_list<std::string_view> types) {
  for (auto t : types) {
    if (t == "f16") {
      s << "enable f16;\n\n";
      return;
    }
  }
}

// Emit a WGSL elem_to_loc function for strided indexing
inline void emit_elem_to_loc(
    std::ostringstream& s,
    const std::string& func_name,
    const std::string& shape_accessor,
    const std::string& stride_accessor) {
  s << "fn " << func_name << "(idx: u32, ndim: u32) -> i32 {\n"
    << "  var loc: i32 = 0;\n"
    << "  var idx_rem = idx;\n"
    << "  for (var d: u32 = ndim - 1u; d < ndim; d = d - 1u) {\n"
    << "    let s = " << shape_accessor << "(d);\n"
    << "    let st = " << stride_accessor << "(d);\n"
    << "    loc += i32(idx_rem % s) * st;\n"
    << "    idx_rem /= s;\n"
    << "  }\n"
    << "  return loc;\n"
    << "}\n\n";
}

// Fill vec4 pairs for shape and stride params
inline void fill_vec4_params(
    uint32_t* shape_0, uint32_t* shape_1,
    int32_t* strides_0, int32_t* strides_1,
    const std::vector<int32_t>& shape,
    const std::vector<int64_t>& strides,
    uint32_t ndim) {
  for (uint32_t i = 0; i < ndim && i < MAX_NDIM; ++i) {
    if (i < 4) {
      shape_0[i] = static_cast<uint32_t>(shape[i]);
      strides_0[i] = static_cast<int32_t>(strides[i]);
    } else {
      shape_1[i - 4] = static_cast<uint32_t>(shape[i]);
      strides_1[i - 4] = static_cast<int32_t>(strides[i]);
    }
  }
}

// Emit an unrolled tree reduction over a shared-memory array.
// The reduce_op should be a WGSL binary function name, e.g. "reduce_op", "max".
inline void emit_unrolled_reduction(
    std::ostringstream& s,
    const std::string& shared_name,
    const std::string& reduce_op,
    uint32_t workgroup_size = WORKGROUP_SIZE) {
  for (uint32_t stride = workgroup_size / 2; stride >= 1; stride /= 2) {
    s << "  if (tid < " << stride << "u) {\n"
      << "    " << shared_name << "[tid] = " << reduce_op << "("
      << shared_name << "[tid], " << shared_name << "[tid + " << stride << "u]);\n"
      << "  }\n  workgroupBarrier();\n";
  }
}

// NOTE: effective_subgroup_size() and subgroup_suffix() call device(), which
// is declared in "mlx/backend/webgpu/device.h". Because these are `inline`
// their bodies are only parsed at instantiation — every translation unit
// that calls them already includes device.h (same pattern as the existing
// create_uniform_buffer / create_bind_group inlines above, which call
// `device().gpu_device()` without utils.h itself including device.h).

// Returns the subgroup_size to use in WGSL emit helpers. 0 = disabled
// (either the device has no Subgroups feature, the probe kernel failed,
// or the device reports a varying size across invocations). Callers that
// see 0 MUST take the tree-reduction path.
//
// The conservative "min != max → disabled" policy is deliberate: this
// helper only returns a non-zero value when the device guarantees a
// single fixed subgroup size for all invocations, which is the contract
// emit_subgroup_reduction's constant-folded WGSL assumes.
inline int effective_subgroup_size() {
  auto& dev = device();
  if (!dev.has_subgroups()) return 0;
  uint32_t lo = dev.subgroup_size_min();
  uint32_t hi = dev.subgroup_size_max();
  if (lo == 0 || hi == 0 || lo != hi) return 0;
  return static_cast<int>(lo);
}

// Append to pipeline-cache entry names when the kernel's WGSL depends on
// the detected subgroup size (i.e. emits subgroup builtins). Returns the
// empty string when the subgroup path is disabled.
//
// Every call site that selects the subgroup path MUST append this suffix
// (not a bare "_sg") to its entry_name, because a cached pipeline compiled
// for subgroup_size=32 must NOT be reused on a device where we later
// detected subgroup_size=64 — the tree-reduction tail in the emitted WGSL
// is constant-folded for a specific lane count.
inline std::string subgroup_suffix() {
  int sg = effective_subgroup_size();
  if (sg == 0) return "";
  return "_sg" + std::to_string(sg);
}

// Emit a subgroup-accelerated reduction.
// Uses subgroup intrinsics for the bottom levels, then a small tree
// reduction across subgroup results in shared memory.
//
// subgroup_builtin: e.g. "subgroupAdd", "subgroupMax", "subgroupMin"
// acc_var:          the per-thread accumulator variable name
// shared_name:      the workgroup shared array
// reduce_op:        WGSL binary function name for the tree phase
// subgroup_size:    the device-detected subgroup size from
//                   effective_subgroup_size(). MUST match the actual runtime
//                   subgroupSize on the device the pipeline executes on —
//                   otherwise the tree-reduction step reads uninitialized
//                   slots in `shared_name` and returns garbage. 32 is only
//                   correct on Apple Silicon / NVIDIA / WARP-32 hardware;
//                   AMD is 64 and Intel is 8/16/32. Callers MUST pass the
//                   detected size explicitly (no default), and MUST cache
//                   the resulting pipeline under a key that includes
//                   subgroup_suffix() so a pipeline compiled for one lane
//                   count is never reused on a device with a different one.
//   num_subgroups is emitted as a constant, so the WGSL is the same number
//   of tree-reduce iterations regardless of device — the pipeline must be
//   cache-keyed per subgroup_size (see subgroup_suffix() above).
inline void emit_subgroup_reduction(
    std::ostringstream& s,
    const std::string& acc_var,
    const std::string& shared_name,
    const std::string& subgroup_builtin,
    const std::string& reduce_op,
    uint32_t workgroup_size,
    uint32_t subgroup_size,
    const std::string& prefix = "") {
  uint32_t num_subgroups = workgroup_size / subgroup_size;

  // Use prefix to avoid variable name collisions when called multiple times
  // in the same shader (e.g., softmax has both max and sum reductions).
  std::string sg_var = prefix.empty() ? "sg_result" : prefix + "_sg_result";
  std::string sg_id = prefix.empty() ? "subgroup_id" : prefix + "_subgroup_id";

  // Step 1: subgroup-level reduction
  s << "  // Subgroup reduction\n"
    << "  var " << sg_var << " = " << subgroup_builtin << "(" << acc_var << ");\n"
    << "  let " << sg_id << " = tid / " << subgroup_size << "u;\n"
    << "  if (subgroupElect()) { " << shared_name << "[" << sg_id << "] = " << sg_var << "; }\n"
    << "  workgroupBarrier();\n";

  // Step 2: tree-reduce across subgroup results (only num_subgroups values)
  for (uint32_t stride = num_subgroups / 2; stride >= 1; stride /= 2) {
    s << "  if (tid < " << stride << "u) {\n"
      << "    " << shared_name << "[tid] = " << reduce_op << "("
      << shared_name << "[tid], " << shared_name << "[tid + " << stride << "u]);\n"
      << "  }\n  workgroupBarrier();\n";
  }
}

} // namespace mlx::core::wgpu
