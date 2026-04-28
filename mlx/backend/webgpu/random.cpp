// Copyright 2026 Apple Inc.
//
// WebGPU RandomBits implementation using Threefry 2x32 PRNG.
//
// The Threefry-2x32-20 algorithm is a counter-based PRNG that produces
// two uint32 outputs per (key, counter) pair. This kernel parallelizes
// over the counter space, with each thread generating one pair.
//
// Reference: Random123: a Library of Counter-Based Random Number Generators
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

#include "mlx/backend/common/utils.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// Uniform buffer layout for RandomBits kernel.
// SPEC-F032: keys are passed as a storage buffer binding (not host-read)
// so the kernel works correctly when keys live GPU-side (e.g. after
// `random.split`). We plumb element offsets into the key buffer so this
// supports both row-contiguous and strided key arrays.
struct RandomBitsParams {
  uint32_t key0_offset;  // element index into keys storage buffer for key0
  uint32_t key1_offset;  // element index into keys storage buffer for key1
  uint32_t half_size;    // number of threads (each produces 2 u32 outputs)
  uint32_t odd;          // 1 if out_per_key is odd, 0 otherwise
  uint32_t bytes_per_key;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};

// Generate the WGSL kernel for random bits generation.
// The kernel implements Threefry-2x32-20 (20 rounds, 2x32-bit).
//
// For simplicity and correctness, we handle one key at a time from the
// host side (the common case is num_keys=1). The kernel writes output
// as bytes packed into u32 words, matching the Metal kernel's byte-level
// output semantics.
std::string make_random_bits_kernel(const std::string& entry_name) {
  std::ostringstream s;

  s << "const WORKGROUP_SIZE: u32 = " << wgpu::WORKGROUP_SIZE << "u;\n\n";

  s << R"(
struct RandomBitsParams {
  key0_offset: u32,
  key1_offset: u32,
  half_size: u32,
  odd: u32,
  bytes_per_key: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: RandomBitsParams;
@group(0) @binding(2) var<storage, read> keys: array<u32>;

// Threefry 2x32 hash function (20 rounds).
// Rotation constants for 2x32: two sets of 4 rotations each,
// alternating every 4 rounds.
fn threefry2x32(key0: u32, key1: u32, ctr0: u32, ctr1: u32) -> vec2<u32> {
  let ks2 = key0 ^ key1 ^ 0x1BD11BDAu;

  // Key schedule array: ks[0]=key0, ks[1]=key1, ks[2]=ks2
  // Rotation constants
  let rot_0 = array<u32, 4>(13u, 15u, 26u, 6u);
  let rot_1 = array<u32, 4>(17u, 29u, 16u, 24u);

  var v0 = ctr0 + key0;
  var v1 = ctr1 + key1;

  // 5 groups of 4 rounds each = 20 rounds total
  // Round group 0 (rotations from rot_0)
  v0 = v0 + v1; v1 = (v1 << rot_0[0]) | (v1 >> (32u - rot_0[0])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[1]) | (v1 >> (32u - rot_0[1])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[2]) | (v1 >> (32u - rot_0[2])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[3]) | (v1 >> (32u - rot_0[3])); v1 = v1 ^ v0;
  // Key injection after round group 0: ks[(0+1)%3], ks[(0+2)%3] + 0+1
  v0 = v0 + key1;
  v1 = v1 + ks2 + 1u;

  // Round group 1 (rotations from rot_1)
  v0 = v0 + v1; v1 = (v1 << rot_1[0]) | (v1 >> (32u - rot_1[0])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_1[1]) | (v1 >> (32u - rot_1[1])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_1[2]) | (v1 >> (32u - rot_1[2])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_1[3]) | (v1 >> (32u - rot_1[3])); v1 = v1 ^ v0;
  // Key injection after round group 1: ks[(1+1)%3], ks[(1+2)%3] + 1+1
  v0 = v0 + ks2;
  v1 = v1 + key0 + 2u;

  // Round group 2 (rotations from rot_0)
  v0 = v0 + v1; v1 = (v1 << rot_0[0]) | (v1 >> (32u - rot_0[0])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[1]) | (v1 >> (32u - rot_0[1])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[2]) | (v1 >> (32u - rot_0[2])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[3]) | (v1 >> (32u - rot_0[3])); v1 = v1 ^ v0;
  // Key injection after round group 2: ks[(2+1)%3], ks[(2+2)%3] + 2+1
  v0 = v0 + key0;
  v1 = v1 + key1 + 3u;

  // Round group 3 (rotations from rot_1)
  v0 = v0 + v1; v1 = (v1 << rot_1[0]) | (v1 >> (32u - rot_1[0])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_1[1]) | (v1 >> (32u - rot_1[1])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_1[2]) | (v1 >> (32u - rot_1[2])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_1[3]) | (v1 >> (32u - rot_1[3])); v1 = v1 ^ v0;
  // Key injection after round group 3: ks[(3+1)%3], ks[(3+2)%3] + 3+1
  v0 = v0 + key1;
  v1 = v1 + ks2 + 4u;

  // Round group 4 (rotations from rot_0)
  v0 = v0 + v1; v1 = (v1 << rot_0[0]) | (v1 >> (32u - rot_0[0])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[1]) | (v1 >> (32u - rot_0[1])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[2]) | (v1 >> (32u - rot_0[2])); v1 = v1 ^ v0;
  v0 = v0 + v1; v1 = (v1 << rot_0[3]) | (v1 >> (32u - rot_0[3])); v1 = v1 ^ v0;
  // Key injection after round group 4: ks[(4+1)%3], ks[(4+2)%3] + 4+1
  v0 = v0 + ks2;
  v1 = v1 + key0 + 5u;

  return vec2<u32>(v0, v1);
}

)";

  // Main compute kernel.
  // Each thread at global index `gid` generates one Threefry pair and writes
  // two u32 values to the output buffer, matching the Metal kernel's layout:
  //   - First value at word index (gid)
  //   - Second value at word index (gid + half_size + odd)
  // When the output has an odd number of u32 words, the last thread in the
  // first half (gid == half_size) only writes one value.
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let tid = gid.x;\n"
    << "  let half_size = params.half_size;\n"
    << "  let odd = params.odd;\n"
    << "  let total_threads = half_size + odd;\n"
    << "\n"
    << "  if (tid >= total_threads) { return; }\n"
    << "\n"
    << "  // Load the key pair from GPU memory. Avoids the stale-host-read\n"
    << "  // bug (SPEC-F032) when the key array was produced on GPU.\n"
    << "  let key0 = keys[params.key0_offset];\n"
    << "  let key1 = keys[params.key1_offset];\n"
    << "\n"
    << "  let drop_last = (odd != 0u) && (tid == half_size);\n"
    << "  let ctr1 = select(tid + total_threads, 0u, drop_last);\n"
    << "  let bits = threefry2x32(key0, key1, tid, ctr1);\n"
    << "\n"
    << "  // Write first u32 output\n"
    << "  output[tid] = bits.x;\n"
    << "\n"
    << "  // Write second u32 output (unless this is the extra odd thread)\n"
    << "  if (!drop_last) {\n"
    << "    output[ctr1] = bits.y;\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;

  // Allocate output buffer.
  // The output is always written as u32 words, so we need enough space
  // for ceil(bytes_per_key / 4) u32 elements per key.
  out.set_data(allocator::malloc(out.nbytes()));

  if (out.size() == 0) {
    return;
  }

  size_t out_per_key = (bytes_per_key + 3) / 4;  // number of u32 words per key
  size_t half_size = out_per_key / 2;
  bool odd = out_per_key % 2;

  auto& s = stream();
  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  // Cache shader by name
  std::string entry_name = "random_bits";
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_random_bits_kernel(entry_name); });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  encoder.set_input_array(keys);
  encoder.set_output_array(out);

  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t out_buf_size = wgpu::wgpu_bind_size(out);
  WGPUBuffer keys_buf = wgpu::wgpu_buffer(keys);
  uint64_t keys_buf_size = wgpu::wgpu_bind_size(keys);

  auto& pool = dev.uniform_pool();

  // Process each key. In the common case (num_keys=1), this loop runs once.
  for (size_t k = 0; k < num_keys; ++k) {
    size_t kidx = 2 * k;

    // SPEC-F032: Compute element offsets into the keys storage buffer
    // instead of dereferencing host-side. This keeps random generation
    // correct when keys live GPU-side (e.g. after `random.split`) and
    // avoids a hidden GPU->CPU synchronization.
    //
    // Convert the keys array's byte offset to an element offset and add
    // the per-key element index. For non-contiguous key arrays we walk
    // the strides via elem_to_loc.
    uint64_t keys_elem_base = static_cast<uint64_t>(keys.offset()) /
        static_cast<uint64_t>(keys.itemsize());
    uint64_t k0_elem, k1_elem;
    if (keys.flags().row_contiguous) {
      k0_elem = keys_elem_base + static_cast<uint64_t>(kidx);
      k1_elem = keys_elem_base + static_cast<uint64_t>(kidx + 1);
    } else {
      auto k0_rel = elem_to_loc(
          static_cast<int>(kidx), keys.shape(), keys.strides());
      auto k1_rel = elem_to_loc(
          static_cast<int>(kidx + 1), keys.shape(), keys.strides());
      k0_elem = keys_elem_base + static_cast<uint64_t>(k0_rel);
      k1_elem = keys_elem_base + static_cast<uint64_t>(k1_rel);
    }

    RandomBitsParams params{};
    params.key0_offset = static_cast<uint32_t>(k0_elem);
    params.key1_offset = static_cast<uint32_t>(k1_elem);
    params.half_size = static_cast<uint32_t>(half_size);
    params.odd = odd ? 1u : 0u;
    params.bytes_per_key = static_cast<uint32_t>(bytes_per_key);

    WGPUBuffer uniform_buf =
        pool.acquire(dev.gpu_queue(), &params, sizeof(RandomBitsParams));

    // For multi-key case, we need to offset into the output buffer.
    // Since WebGPU bind groups use (buffer, offset, size), we'd ideally
    // use dynamic offsets. For simplicity, bind the whole buffer and let
    // the shader handle the base offset.
    //
    // However, the current shader writes starting at index 0.
    // For num_keys > 1, we would need a different approach (add base_offset
    // to params, or use a 2D dispatch like Metal).
    //
    // Since the overwhelmingly common case is num_keys=1 (single key for
    // sampling), and multi-key support would require additional shader
    // complexity, we handle num_keys=1 efficiently and throw for multi-key
    // cases until needed.
    if (num_keys > 1) {
      pool.release(uniform_buf);
      throw std::runtime_error(
          "[RandomBits::eval_gpu] WebGPU backend currently supports only "
          "single-key random generation. Got " +
          std::to_string(num_keys) + " keys.");
    }

    WGPUBindGroup bg = wgpu::create_bind_group(
        pe.layout,
        {{out_buf, out_buf_size},
         {uniform_buf, sizeof(RandomBitsParams)},
         {keys_buf, keys_buf_size}});

    uint32_t total_threads = static_cast<uint32_t>(half_size + (odd ? 1 : 0));
    uint32_t num_workgroups =
        (total_threads + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
    encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

    wgpuBindGroupRelease(bg);
    encoder.add_completed_handler(
        [uniform_buf]() { wgpu::device().uniform_pool().release(uniform_buf); });
  }
}

} // namespace mlx::core
