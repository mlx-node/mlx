# WebGPU Backend Architecture

The WebGPU backend enables MLX to run on any platform with a WebGPU implementation — including browsers (via WASM), desktop (via Dawn or wgpu-native), and embedded systems. This document covers the full architecture: the C++ kernel layer, the browser bridge, the WASM build system, and the pitfalls we encountered along the way.

## Table of Contents

- [Overview](#overview)
- [Two-Worker Architecture (Browser)](#two-worker-architecture-browser)
- [C++ Backend Architecture](#c-backend-architecture)
- [Type System: CPU vs GPU Widths](#type-system-cpu-vs-gpu-widths)
- [Uniform Buffer Pooling](#uniform-buffer-pooling)
- [Memory Management](#memory-management)
- [WASM Build System](#wasm-build-system)
- [Pitfalls and Bugs Found](#pitfalls-and-bugs-found)
- [Performance Characteristics](#performance-characteristics)
- [Operations Not Yet Implemented](#operations-not-yet-implemented)
- [Adding a New Kernel](#adding-a-new-kernel)
- [Debugging Tips](#debugging-tips)

---

## Overview

The backend is split across two repositories:

- **mlx** (`mlx/backend/webgpu/`) — C++ GPU kernels that generate WGSL shader source at runtime, compile and cache pipelines, and dispatch compute work via the standard `webgpu.h` C API.
- **mlx-node** — TypeScript bridge that implements `webgpu.h` functions as JavaScript calls into the browser's `GPUDevice`, connected to the WASM-compiled MLX via SharedArrayBuffer + Atomics RPC.

Three `webgpu.h` implementations are supported:

| Backend       | Description                                    |
|---------------|------------------------------------------------|
| `DAWN`        | Google's reference WebGPU (desktop)            |
| `WGPU`        | wgpu-native pre-built library (desktop)        |
| `WASI_IMPORT` | WASM target — functions imported from JS       |

For the browser target (`WASI_IMPORT`), all `wgpu*` function calls become unresolved WASM imports, satisfied at runtime by the TypeScript bridge in mlx-node.

---

## Two-Worker Architecture (Browser)

The browser deployment uses a two-worker architecture to work around the constraint that WebGPU's `GPUDevice` can only be used on the thread that created it, while WASM needs synchronous blocking (`Atomics.wait`) that is forbidden on the main thread:

```
Main Thread (UI)          wasm-worker              gpu-worker
    |                        |                         |
    | <- postMessage <-      | <- SharedArrayBuffer -> |
    |   (results)            |   (Atomics RPC)         |
    |                        |                         |
    | <- Atomics.waitAsync < | -> Atomics.wait ->      |
    |   (stream tokens)      |   (blocks for GPU)      | <- GPUDevice
```

**wasm-worker** — Runs the compiled MLX WASM module. When MLX C++ code calls a WebGPU function (e.g., `wgpuDeviceCreateComputePipeline`), the bridge stub encodes the call into a SharedArrayBuffer command region and wakes the gpu-worker via `Atomics.notify`. Then it blocks on `Atomics.wait` until the gpu-worker writes back the result.

**gpu-worker** — Owns the `GPUDevice`. Sits in a `Atomics.waitAsync` loop. When woken, it reads the command from SharedArrayBuffer, executes the corresponding WebGPU API call, writes the result back, and notifies the wasm-worker.

### SharedArrayBuffer RPC Protocol

The command buffer is a fixed-size (512-byte) SharedArrayBuffer with this layout:

```
Offset  Field              Size     Description
------  -----              ----     -----------
0       CMD_FN_ID          4 bytes  RPC function ID
4       CMD_STATE          4 bytes  Atomics wait/notify flag (Int32Array)
8       CMD_RESULT         8 bytes  Return value (lo + hi)
16      CMD_ARG0..ARG7     32 bytes Up to 8 u32 arguments
48      ARG0_HI..ARG3_HI   16 bytes High bits for u64 args
64      CALLBACK_COUNT     4 bytes  Pending callback count
68      CALLBACK_DATA      120 bytes Callback payloads (8 entries x 16 bytes)
188     UNIFORM_DATA_SIZE  4 bytes  Inline uniform buffer size
192     UNIFORM_DATA       256 bytes Inline uniform buffer data
448     Reserved           64 bytes
```

Each RPC call:

1. wasm-worker writes `FN_ID` and `ARG0..ARGn`
2. wasm-worker sets `CMD_STATE = PENDING` and calls `Atomics.notify`
3. wasm-worker calls `Atomics.wait(CMD_STATE, PENDING)` (blocks)
4. gpu-worker wakes, reads command, executes, writes `CMD_RESULT`
5. gpu-worker sets `CMD_STATE = DONE` and calls `Atomics.notify`
6. wasm-worker wakes and reads the result

**Fused dispatch optimization**: To reduce RPC round-trips, the bridge batches `setPipeline` + `setBindGroup` + `dispatchWorkgroups` into a single `FUSED_DISPATCH` RPC call (function ID 91). The `FUSED_DISPATCH_WITH_UNIFORM` variant (ID 97) additionally creates the uniform buffer and writes data inline — turning what would be 5+ RPCs (createBuffer, getMappedRange, unmap, setPipeline, setBindGroup, dispatch) into 1.

### Handle Management (gpu-worker)

GPU objects (buffers, pipelines, bind groups, etc.) are tracked by integer handles in a sparse array on the gpu-worker side. When the wasm-worker calls `wgpuDeviceCreateBuffer`, the gpu-worker creates the real `GPUBuffer`, stores it in the handle table, and returns the integer handle. All subsequent references use this handle.

### Weight Upload Flow

1. mlx-worker downloads SafeTensors weights into a `SharedArrayBuffer`
2. `postMessage({ type: 'upload_weights', weightsSab, tensors })` to gpu-worker
3. gpu-worker creates GPU buffers with `mappedAtCreation: true`
4. bf16 weights: expanded to f32 inline (`dst32[j] = src16[j] << 16`)
5. f16 weights: IEEE 754 conversion to f32 (or kept as f16 if `shader-f16` available)
6. `gpuBuffer.unmap()`, return handle to wasm-worker
7. wasm-worker passes handles to C++ via `mlx_array_from_gpu_buffer()` FFI

### Token Streaming

Decoded tokens are streamed to the main thread via a separate SharedArrayBuffer channel:

```
[0..3]   u32: byte length of accumulated text
[4..7]   u32: sequence counter (incremented per token)
[8..N]   utf-8 text data
```

The WASM module writes tokens via `mlx_stream_write()`, increments the sequence counter with `Atomics.add`, and wakes the main thread with `Atomics.notify`. The main thread uses `Atomics.waitAsync` for non-blocking per-token rendering.

---

## C++ Backend Architecture

All kernel code lives in `mlx/backend/webgpu/`. Each `.cpp` file typically implements one or more MLX primitives by:

1. Defining a C++ params struct (uniform buffer layout)
2. Generating WGSL shader source as a string
3. Caching the compiled pipeline via `device().get_or_create_shader_module()`
4. Setting up bind groups and dispatching compute work

### Source Files

| File               | Operations                                                  |
|--------------------|-------------------------------------------------------------|
| `unary.cpp`        | 46 ops: Abs, Exp, Log, Sin, Cos, Sqrt, Sigmoid, Erf, etc.  |
| `binary.cpp`       | 18 ops: Add, Multiply, Power, Maximum, comparisons, etc.    |
| `reduce.cpp`       | Sum, Prod, Max, Min, And, Or (3 strategies)                 |
| `matmul.cpp`       | GEMV (per-element) and GEMM (tiled 16x16)                   |
| `quantized.cpp`    | QuantizedMatmul (int2/int4/int8 on-the-fly dequant)         |
| `softmax.cpp`      | Online softmax (3-phase: max, sum-exp, normalize)           |
| `normalization.cpp`| RMSNorm (2-phase) and LayerNorm (3-phase)                   |
| `rope.cpp`         | RoPE: single-token and general variants                     |
| `scan.cpp`         | Prefix sum/prod/max/min/logaddexp                           |
| `sort.cpp`         | Bitonic sort and argsort in shared memory                   |
| `conv.cpp`         | Depthwise and grouped conv1d                                |
| `indexing.cpp`     | Gather, GatherAxis, Scatter, ScatterAxis, SliceUpdate       |
| `copy.cpp`         | Copy with 6 dtype conversion modes                          |
| `arange.cpp`       | Integer and float range generation                          |
| `arg_reduce.cpp`   | ArgMax, ArgMin                                              |
| `ternary.cpp`      | Select (where/conditional)                                  |
| `logsumexp.cpp`    | LogSumExp (2-phase: max, log-sum-exp)                       |
| `random.cpp`       | RandomBits via Threefry-2x32-20 PRNG                        |
| `slicing.cpp`      | Concatenate, dynamic slice offset                           |
| `device.cpp`       | Device init, feature detection, caching                     |
| `allocator.cpp`    | Buffer allocation, page cache, readback                     |
| `utils.h`          | Type helpers, bind group creation, WGSL codegen             |

### WGSL Code Generation Pattern

Every kernel follows the same pattern. Here is a simplified example for a unary operation:

```cpp
std::string make_unary_kernel(
    const std::string& entry_name,
    const std::string& in_type,
    const std::string& op_expr) {
  std::ostringstream s;

  if (in_type == "f16") s << "enable f16;\n";

  s << "const WORKGROUP_SIZE: u32 = 256u;\n"
    << "const N_READS: u32 = 4u;\n\n"
    << "@group(0) @binding(0) var<storage, read> input: array<" << in_type << ">;\n"
    << "@group(0) @binding(1) var<storage, read_write> output: array<" << in_type << ">;\n"
    << "@group(0) @binding(2) var<uniform> params: Params;\n\n"
    << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let base = gid.x * N_READS;\n"
    << "  for (var i = base; i < min(base + N_READS, params.size); i++) {\n"
    << "    let in_val = input[i + params.in_offset];\n"
    << "    output[i + params.out_offset] = " << op_expr << ";\n"
    << "  }\n}\n";

  return s.str();
}
```

Kernels are cached by name:

```cpp
std::string entry = "abs_f32_v";
WGPUShaderModule shader = dev.get_or_create_shader_module(
    entry, [&]() { return make_unary_kernel(entry, "f32", "abs(in_val)"); });
auto pe = dev.get_or_create_pipeline(entry, shader, entry.c_str());
```

Each kernel variant gets a unique name encoding the dtype, operation, and variant (`v` for contiguous, `g` for general/strided, `ss`/`sv`/`vs`/`vv` for binary scalar/vector combinations).

### Uniform Buffer Layout

All uniform parameters must be vec4-aligned (16 bytes). A typical params struct:

```cpp
struct UnaryParams {
  uint32_t size_ndim[4];   // [size, ndim, pad, pad]
  uint32_t offsets[4];     // [in_offset, out_offset, pad, pad]
  uint32_t shape_0[4];     // shape[0..3]
  uint32_t shape_1[4];     // shape[4..7]
  int32_t strides_0[4];    // in_strides[0..3]
  int32_t strides_1[4];    // in_strides[4..7]
};
```

Shape and strides are split across two vec4s to support up to `MAX_NDIM = 8` dimensions. The WGSL side accesses them via helper functions like `get_shape(i)` and `get_stride(i)` that index into the appropriate vec4.

### Reduction Strategy

Reductions use one of two algorithms depending on device capabilities:

**Tree reduction** (fallback) — The `emit_unrolled_reduction` helper generates an unrolled shared-memory tree:

```wgsl
// Generated WGSL for workgroup_size=256:
if (tid < 128u) { shared[tid] = op(shared[tid], shared[tid + 128u]); }
workgroupBarrier();
if (tid < 64u) { shared[tid] = op(shared[tid], shared[tid + 64u]); }
workgroupBarrier();
// ... down to stride 1
```

**Subgroup reduction** (when `device().has_subgroups()`) — The `emit_subgroup_reduction` helper emits a two-phase pattern:

```wgsl
enable subgroups;
var sg_val = subgroupAdd(acc);           // Phase 1: hardware reduction
if (subgroupElect()) {
  shared[subgroup_id] = sg_val;          // One value per subgroup
}
workgroupBarrier();
// Phase 2: tree-reduce across ~8 subgroup results
if (tid < 4u) { shared[tid] = op(shared[tid], shared[tid + 4u]); }
workgroupBarrier();
// ... down to stride 1
```

The `prefix` parameter avoids WGSL variable name collisions when a kernel needs multiple reductions (e.g., softmax uses "mx" for max and "sm" for sum).

---

## Type System: CPU vs GPU Widths

A central challenge of the WebGPU backend is that some MLX dtypes have different sizes on CPU and GPU:

| Dtype    | CPU size | GPU type       | GPU size |
|----------|----------|----------------|----------|
| float32  | 4 bytes  | `f32`          | 4 bytes  |
| float16  | 2 bytes  | `f16` or `f32` | 2 or 4   |
| bfloat16 | 2 bytes  | `f32`          | 4 bytes  |
| bool     | 1 byte   | `u32`          | 4 bytes  |
| int32    | 4 bytes  | `i32`          | 4 bytes  |
| uint32   | 4 bytes  | `u32`          | 4 bytes  |

This mismatch requires careful handling throughout:

**Allocation**: `wgpu_alloc_size(arr)` computes GPU buffer size using `wgpu_itemsize()` which returns 4 for bool and bfloat16.

**Upload conversion**: When data is uploaded to the GPU (`device.cpp` `upload_with_conversion`), bfloat16 values are expanded to float32 (`bf16_bits << 16`) and bool values are expanded to uint32.

**Download conversion**: When data is read back (`allocator.cpp` `raw_ptr()`), float32 values are truncated back to bfloat16 and uint32 values are truncated to bool.

**Offset calculation**: Array offsets are always in bytes on the CPU side. To get the GPU element index:

```cpp
uint32_t gpu_elem_idx = arr.offset() / arr.itemsize();  // bytes -> elements
```

This works because each CPU element maps 1:1 to a GPU element, regardless of size difference.

**`dtype_to_wgsl_safe()`**: Returns `"f32"` instead of `"f16"` when the device lacks `shader-f16` support, ensuring graceful fallback.

---

## Uniform Buffer Pooling

Creating and destroying uniform buffers per dispatch is expensive. The `UniformBufferPool` class in `device.h` manages a free list of reusable buffers, organized by 256-byte-aligned size:

1. `pool.acquire(queue, data, size)` — finds a free buffer of the right size (or creates one), writes data via `wgpuQueueWriteBuffer`
2. After GPU work completes, the buffer is returned via `encoder.add_completed_handler([buf]() { pool.release(buf); })`

---

## Memory Management

The `WebGPUAllocator` uses a `BufferCache` with 16 KB page granularity. Each allocation is represented by a `WebGPUBuffer` struct:

```cpp
struct WebGPUBuffer {
  WGPUBuffer buffer;       // GPU-side storage
  size_t size;             // Allocated bytes
  void* cpu_ptr;           // Non-null after readback
  bool cpu_dirty;          // CPU data not yet uploaded
  bool gpu_has_data;       // GPU buffer has meaningful data
  Dtype::Val dtype_val;    // For conversion on readback
};
```

The `cpu_ptr` field is lazily allocated on first `raw_ptr()` call (GPU readback). It is **invalidated** after GPU writes to prevent stale data from being returned — this was a critical bug fix (see [Pitfalls](#stale-cpu_ptr-after-gpu-compute)).

---

## WASM Build System

The WASM build involves multiple stages:

1. **Cargo + NAPI-RS** compiles Rust code targeting `wasm32-wasip1-threads`
2. **cmake** (invoked by `build.rs`) cross-compiles MLX C++ with WASI-SDK
3. **wasm-ld** links everything into a single `.wasm` binary (~23 MB)
4. The binary is copied to `packages/browser/dist/index.wasm`

Key build flags:

```bash
# C++ compilation
-fwasm-exceptions          # WASM native exception handling
-fno-inline                # Prevents vtable method inlining (GC safety)
-femit-all-decls           # Forces emission of inline virtual methods
-fvisibility=hidden        # Consistent vtable visibility

# Linking
-Wl,--whole-archive=mlx    # Prevents vtable method GC
-zstack-size=16777216      # 16 MB stack for deep model call stacks
--export=<526 symbols>     # Force-export virtual methods (eval_gpu_exports.txt)

# cmake
-DMLX_BUILD_WEBGPU=ON
-DWEBGPU_BACKEND=WASI_IMPORT
-DMLX_BUILD_METAL=OFF
-DMLX_BUILD_CUDA=OFF
```

**Critical**: The MLX source at `crates/mlx-sys/mlx` is a **symlink** to the main MLX repo, not a git submodule. All C++ changes are made in the mlx repo and automatically picked up by WASM builds. Never re-init this as a submodule.

**No Asyncify**: The architecture uses `Atomics.wait` for synchronous blocking instead of Binaryen's asyncify transform. This avoids a ~60 MB binary size increase and emnapi corruption issues.

---

## Pitfalls and Bugs Found

This section documents every significant bug and pitfall encountered during development, organized by category. Each entry explains the symptom, root cause, and fix.

### Type Width Mismatches

#### bf16 offset calculation in copy kernel

**Symptom**: bf16 arrays with non-zero offsets read from wrong buffer positions. 2 tests failing.

**Root cause**: The copy kernel computed GPU offsets by dividing byte offsets by 4 (u32 size). For bf16 (CPU itemsize=2, GPU itemsize=4), an element at byte offset 64 became GPU index 16 instead of the correct 32:

```cpp
// WRONG: byte_offset / sizeof(u32)
uint32_t gpu_idx = (arr.offset() + elem * 2) / 4;  // 64/4 = 16

// CORRECT: byte_offset / cpu_itemsize = element_index = gpu_index
uint32_t gpu_idx = arr.offset() / arr.itemsize() + elem;  // 64/2 = 32
```

**Fix**: Convert byte offset to element offset first, then use as GPU index directly (each element = one u32 slot on GPU).

#### Bool/bf16 allocation size

**Symptom**: GPU buffer too small for bool and bf16 arrays, causing out-of-bounds writes.

**Root cause**: Allocation used `arr.nbytes()` (CPU bytes) but GPU needs 4 bytes per element for bool (1 byte on CPU) and bf16 (2 bytes on CPU).

**Fix**: `wgpu_alloc_size()` uses `arr.size() * wgpu_itemsize(dtype)` to compute GPU-sized allocation. `ensure_wgpu_size()` reallocates if the GPU type is wider than the CPU type.

### Buffer Coherence

#### Stale cpu_ptr after GPU compute

**Symptom**: After GPU computation, reading array data returned the original (pre-compute) values instead of GPU results.

**Root cause**: `raw_ptr()` returned the cached `cpu_ptr` without checking whether the GPU had written new data since the last readback.

**Fix**: Invalidate `cpu_ptr` (set to `nullptr`) in `set_output_array()` and after upload in `wgpu_buffer()`. This forces a fresh GPU readback on the next `raw_ptr()` call.

#### WebGPU buffer aliasing violation

**Symptom**: Validation errors when input and output arrays share the same underlying buffer.

**Root cause**: WebGPU requires that a buffer bound as `storage, read` and `storage, read_write` in the same dispatch must be different buffers. Unlike Metal, in-place operations are not allowed.

**Fix**: `ensure_no_alias(out, in)` checks whether two arrays share a buffer and forces a fresh allocation for the output if they do. Applied in all kernels that could potentially operate in-place (RoPE, copy, etc.).

### Kernel Correctness

#### QuantizedMatmul transposed weight indexing

**Symptom**: Garbage output from all quantized linear projections. Model produced random tokens.

**Root cause**: The WGSL kernel had separate branches for transposed vs. non-transposed weights. The transposed branch used `w[packed_idx * N + n]` assuming column-major layout, but quantized weights are **always** stored row-major as `[N, K_packed]`:

```c
// WRONG: assumed column-major for transposed case
w[packed_idx * N + n]

// CORRECT: weight is always [N, K_packed] row-major
w[n * w_cols + packed_idx]
```

**Fix**: Always use row-major indexing regardless of transpose flag.

#### RMSNorm/LayerNorm offset bug

**Symptom**: Garbage normalization output for arrays that are views into larger buffers (non-zero offset).

**Root cause**: The contiguity check was:

```cpp
!x.flags().contiguous || x.strides()[x.ndim() - 1] != 1
```

This allowed arrays with non-zero offsets through. The kernel always reads from buffer position 0, ignoring the array's actual data offset.

**Fix**: Changed to `!x.flags().row_contiguous || x.offset() != 0`, which forces a contiguous copy (resetting offset to 0) for any array with a non-zero offset.

#### Subgroup reduction WGSL name collisions

**Symptom**: WGSL shader compilation failure when a kernel uses subgroup reductions twice (e.g., softmax needs both max and sum reductions).

**Root cause**: `emit_subgroup_reduction` generated the same variable names (`sg_result`, `subgroup_id`) for both reductions, causing WGSL redeclaration errors.

**Fix**: Added a `prefix` parameter to `emit_subgroup_reduction`. Softmax uses "mx" for the max reduction and "sm" for the sum reduction.

#### Matmul missing offset support

**Symptom**: Incorrect results when matmul inputs are views into larger buffers (e.g., sliced attention heads).

**Root cause**: The GEMV and GEMM kernels computed buffer base as `batch * batch_stride` with no per-array offset. Arrays with non-zero offsets read from the wrong position.

**Fix**: Added `offset_a` and `offset_b` fields to `MatmulParams`, computed as `arr.offset() / arr.itemsize()`. Kernels add the offset to the batch base.

### Build System

#### WASM vtable method garbage collection

**Symptom**: Runtime "function signature mismatch" errors in WASM. Virtual method calls crash because the target function was removed by the linker.

**Root cause**: `wasm-ld`'s garbage collection discards functions whose only references are vtable DATA relocations (not CODE relocations). MLX's `eval_gpu` virtual methods are only called through vtable dispatch, so the linker thinks they're unused.

**Fix**: Force-export 526 virtual method symbols via `--export=<symbol>` linker flags, generated from `eval_gpu_exports.txt`. Combined with `-Wl,--whole-archive=mlx` to prevent any archive member from being discarded.

#### LTO corrupts C++ vtables in WASM

**Symptom**: Random crashes and wrong function dispatch after enabling LTO.

**Root cause**: Link-time optimization across the Rust/C++ boundary can merge or reorder vtable entries incorrectly.

**Fix**: `lto = false` in the WASM build profile. LTO remains enabled for native builds.

#### cmake cache invalidation for WGSL changes

**Symptom**: WGSL kernel changes not picked up after rebuild.

**Root cause**: WGSL files are embedded via `file(READ)` at cmake configure time, not build time. Changing a `.wgsl` file doesn't trigger reconfiguration.

**Fix**: Touch `CMakeLists.txt` to force reconfigure. Never clear the cmake build cache entirely (fragile with WASM cross-compilation).

### Browser Environment

#### SharedArrayBuffer requires Cross-Origin-Isolation

`Atomics.wait` and `SharedArrayBuffer` require the page to be served with COOP/COEP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without these, the two-worker architecture cannot function.

#### Atomics.waitAsync requires Int32Array

**Symptom**: `Atomics.waitAsync` silently fails.

**Root cause**: `Atomics.waitAsync` only works with `Int32Array`, not `Uint32Array`. Using the wrong typed array view causes the async wait to never resolve.

#### BigInt in hot paths is very slow

**Symptom**: Unexpectedly low throughput in the TypeScript bridge.

**Root cause**: JavaScript BigInt operations (needed for 64-bit values) are orders of magnitude slower than regular number operations. Using BigInt in the RPC hot path severely impacts throughput.

**Fix**: Use plain 32-bit numbers wherever possible. Split 64-bit values into two 32-bit halves.

---

## Performance Characteristics

As of the latest measurements on Chrome desktop:

| Metric              | Value      |
|---------------------|------------|
| Decode throughput   | 19.2 tok/s |
| Time to first token | 631 ms     |
| Test suite pass rate| 160/163    |

Model: Qwen 3.5 0.8B (bf16, 24 layers, GDN linear attention)

The 3 remaining test failures are all expected:

- `Compiled` primitive not implemented (2 tests) — would require a WGSL JIT compiler for fused element-wise kernels
- Stream channel test (1 test) — expects native environment

---

## Operations Not Yet Implemented

These operations throw or fall back to CPU decomposition:

| Operation                     | Status          | Notes                                          |
|-------------------------------|-----------------|------------------------------------------------|
| `Compiled`                    | NO_GPU (throws) | Graph fusion JIT — major engineering effort    |
| `ScaledDotProductAttention`   | Fallback        | Decomposes to matmul+softmax (works, no flash) |
| `FFT`                         | NO_GPU          | Not needed for transformer inference           |
| `Hadamard`                    | NO_GPU          | Rarely used                                    |
| `BitwiseBinary`               | NO_GPU          | Simple to implement if needed                  |
| `GatherMM`                    | NO_GPU          | Used by some MoE implementations               |
| `GatherQMM`                   | NO_GPU          | Quantized variant of GatherMM                  |
| `Inverse` / `Cholesky`        | NO_GPU          | Linear algebra decompositions                  |
| `SVD` / `Eigh` / `Eig`       | NO_GPU          | Eigenvalue decompositions                      |
| `LUF` / `QRF`                 | NO_GPU          | Matrix factorizations                          |

---

## Adding a New Kernel

To add a new WebGPU kernel implementation:

1. Create `mlx/backend/webgpu/<op_name>.cpp`
2. Define a params struct (vec4-aligned)
3. Write a `make_<op>_kernel()` function that generates WGSL source
4. Implement `<Primitive>::eval_gpu()` following the standard pattern:
   - Ensure inputs are contiguous (or handle strides)
   - Allocate output with `wgpu_alloc_size()`
   - Get or create shader module and pipeline
   - Set up bind group and dispatch
   - Use uniform pool for params, release in completed handler
5. Remove the `NO_GPU` / `NO_GPU_USE_FALLBACK` entry from `primitives.cpp` and add a comment pointing to the new file
6. Add the file to `CMakeLists.txt`
7. If the op was `NO_GPU_USE_FALLBACK`, implement `use_fallback()` returning `false`

Standard dispatch pattern:

```cpp
void MyOp::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& in = inputs[0];

  // 1. Ensure contiguous (if needed)
  array in_contig = in;
  if (!in.flags().row_contiguous || in.offset() != 0) {
    in_contig = contiguous_copy_gpu(in, s);
    auto& encoder = wgpu::get_command_encoder(s);
    encoder.add_temporary(in_contig);
  }

  // 2. Allocate output
  out.set_data(
      allocator::malloc(wgpu::wgpu_alloc_size(in_contig)),
      in_contig.data_size(),
      in_contig.strides(),
      in_contig.flags());

  // 3. Get/create pipeline
  auto& dev = wgpu::device();
  std::string entry = "my_op_" + std::string(wgpu::dtype_to_wgsl_safe(in.dtype()));
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry, [&]() { return make_my_op_kernel(entry, ...); });
  auto pe = dev.get_or_create_pipeline(entry, shader, entry.c_str());

  // 4. Set up params + bind group
  MyOpParams params{};
  params.data[0] = ...;
  auto& pool = dev.uniform_pool();
  WGPUBuffer ubuf = pool.acquire(dev.gpu_queue(), &params, sizeof(params));

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_input_array(in_contig);
  encoder.set_output_array(out);

  WGPUBindGroup bg = wgpu::create_bind_group(pe.layout, {
      {wgpu::wgpu_buffer(in_contig), wgpuBufferGetSize(wgpu::wgpu_buffer(in_contig))},
      {wgpu::wgpu_buffer(out), wgpuBufferGetSize(wgpu::wgpu_buffer(out))},
      {ubuf, sizeof(params)}});

  // 5. Dispatch
  uint32_t n_workgroups = (out.size() + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, n_workgroups);

  // 6. Cleanup
  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([ubuf]() {
    wgpu::device().uniform_pool().release(ubuf);
  });
}
```

---

## Debugging Tips

- **WGSL compilation errors**: The generated WGSL source is a runtime string. Add a `fprintf(stderr, "%s\n", wgsl.c_str())` before `get_or_create_shader_module` to see the exact shader source.

- **Buffer content inspection**: Call `synchronize()` on the stream, then `raw_ptr()` on the array to force a GPU readback. Print the first few elements to verify correctness.

- **Offset bugs**: Always check `arr.offset()` when debugging wrong results. If the kernel doesn't account for offsets, views into larger buffers will read from the wrong location.

- **Type promotion**: Remember that bf16 is stored as f32 on GPU. When comparing GPU output to CPU reference, account for the precision difference (bf16 has ~3 decimal digits of precision).

- **Browser console**: Check for WebGPU validation errors in the browser console. Common issues: buffer aliasing (same buffer in read and read_write slots), buffer too small, workgroup size mismatch.

- **WASM crashes**: Deep model call stacks can overflow the 16 MB WASM stack. Symptoms: silent crash or `RuntimeError: unreachable`. Increase `-zstack-size` in the build script.

- **WASM rebuild not picking up changes**: Touch `CMakeLists.txt` to force cmake reconfigure. Touch `build.rs` to force Cargo to re-run the build script. Never delete the cmake build cache.
