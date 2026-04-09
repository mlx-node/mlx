// Copyright 2026 Apple Inc.
//
// WebGPU fused ScaledDotProductAttention — VECTOR path only (Tq == 1, i.e.
// decode step). The prompt path (Tq > 1) is left on the decomposed fallback.
//
// Algorithm (two-phase, FlashAttention-style but simpler):
//   For each (batch, query_head), one workgroup of WG threads:
//
//     Phase A — parallel reduction over L (sequence length):
//         m = max_l (score_l)          where score_l = dot(Q, K[l]) * scale (+ mask[l])
//         s = sum_l exp(score_l - m)
//
//     Phase B — each thread owns one output lane d = tid (D <= 128):
//         O[d] = sum_l (exp(score_l - m) / s) * V[l, d]
//
// The two-phase variant reads Q, K twice (once in Phase A for scores, once in
// Phase B) but is much easier to reason about than the online single-pass
// variant which would require per-step workgroup broadcasts. Correctness first.
//
// - Workgroup size: 128 threads (matches max supported head dim).
// - One workgroup per (batch * n_heads). Dispatch grid = (B*H, 1, 1).
// - Head dim D is baked into the WGSL source as a `const`, so loops over D
//   unroll and the pipeline cache keys by (dtype, D, has_mask, subgroups).
// - bf16 arrays arrive on the GPU already promoted to f32 (see
//   device.cpp::upload_with_conversion), so we emit f32 WGSL types in both
//   the float32 and bfloat16 paths (via dtype_to_wgsl_safe).
// - GQA: kv_head = h / gqa_factor, gqa_factor = H / H_kv. Both K and V are
//   indexed with kv_head, Q with h.
// - Mask (optional): when `has_arr_mask` is true and the mask is a non-bool
//   array, it's broadcast by `fast.cpp` to the Q output shape [B, H, 1, L].
//   We require it to be row-contiguous and read `mask[(b*H + h)*L + l]`
//   directly. Bool masks stay on the fallback (supports_bool_mask = false,
//   so fast.cpp converts them to additive masks before we get here — but we
//   still return true from use_fallback for safety if anything non-standard
//   sneaks through).
//
// Dispatch layout:
//   group(0) binding(0)  : q           [B, H,     1, D]  (read)
//   group(0) binding(1)  : k           [B, H_kv,  L, D]  (read)
//   group(0) binding(2)  : v           [B, H_kv,  L, D]  (read)
//   group(0) binding(3)  : mask        [B, H,     1, L]  (read, optional)
//   group(0) binding(N)  : output      [B, H,     1, D]  (read_write)
//   group(0) binding(N+1): params      (uniform)
//
// Pipeline cache key: "sdpa_v_" + dtype + "_D" + D + (has_mask ? "_m" : "")
//                     + (has_subgroups ? "_sg" : "").

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/fast_primitives.h"

#include <cassert>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace mlx::core::fast {

namespace {

// SDPA workgroup size: 128 threads. This matches the largest head dim we
// accept (D = 128), so each thread owns one lane of the output when D == 128.
// For D = 64 / 96, threads with tid >= D idle during the per-lane Phase B
// write but still participate in the Phase A reduction.
constexpr uint32_t SDPA_WORKGROUP_SIZE = 128;

// Uniform struct laid out to match the WGSL definition below.
// Total size = 32 bytes (8 u32/f32 slots). Padded to 256 by the WGPU pool.
struct SdpaParams {
  uint32_t B;          // batch
  uint32_t H;          // query heads
  uint32_t H_kv;       // kv heads
  uint32_t L;          // kv sequence length
  uint32_t D;          // head dim (also baked as a WGSL const)
  uint32_t has_mask;   // 0 or 1
  uint32_t gqa_factor; // H / H_kv
  float    scale;      // scale factor (usually 1/sqrt(D), but caller decides)
};

// ---------------------------------------------------------------------------
// WGSL kernel generation
// ---------------------------------------------------------------------------

std::string make_sdpa_vector_kernel(
    const std::string& entry_name,
    const std::string& dtype,
    uint32_t D,
    bool has_mask,
    bool use_subgroups) {
  std::ostringstream s;

  if (use_subgroups) {
    s << "enable subgroups;\n";
  }
  if (dtype == "f16") {
    s << "enable f16;\n";
  }
  s << "\n";

  s << "const WORKGROUP_SIZE: u32 = " << SDPA_WORKGROUP_SIZE << "u;\n";
  s << "const D: u32 = " << D << "u;\n\n";

  s << "struct SdpaParams {\n"
    << "  B: u32,\n"
    << "  H: u32,\n"
    << "  H_kv: u32,\n"
    << "  L: u32,\n"
    << "  D: u32,\n"
    << "  has_mask: u32,\n"
    << "  gqa_factor: u32,\n"
    << "  scale: f32,\n"
    << "}\n\n";

  // Bindings: q, k, v, [mask], output, uniform
  uint32_t binding = 0;
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> q: array<"
    << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> k: array<"
    << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++ << ") var<storage, read> v: array<"
    << dtype << ">;\n";
  if (has_mask) {
    s << "@group(0) @binding(" << binding++
      << ") var<storage, read> mask: array<" << dtype << ">;\n";
  }
  s << "@group(0) @binding(" << binding++
    << ") var<storage, read_write> output: array<" << dtype << ">;\n";
  s << "@group(0) @binding(" << binding++
    << ") var<uniform> params: SdpaParams;\n\n";

  // Workgroup shared memory:
  //   - shared_red  : scratch for Phase A reductions (max then sum, reused)
  //   - shared_scale: Phase A outputs (m, s) broadcast to all threads
  //   - For Phase B we stream scores through shared memory too, reusing
  //     shared_red as "shared_exp" so Phase B doesn't need to recompute scores.
  //     BUT: shared_red has only WORKGROUP_SIZE slots (128), and L can be
  //     much larger, so this optimisation doesn't apply generally. Instead
  //     Phase B recomputes dot(Q, K[l]) — 2x K traffic, but simple.
  //   - shared_q    : Q vector cached once at workgroup start (D floats).
  s << "var<workgroup> shared_red: array<f32, WORKGROUP_SIZE>;\n";
  s << "var<workgroup> shared_m: f32;\n";
  s << "var<workgroup> shared_inv_s: f32;\n";
  s << "var<workgroup> shared_q: array<f32, D>;\n\n";

  // Helper ops for the shared-memory tree reduction.
  s << "fn max_op(a: f32, b: f32) -> f32 { return max(a, b); }\n";
  s << "fn sum_op(a: f32, b: f32) -> f32 { return a + b; }\n\n";

  // Main entry.
  s << "@compute @workgroup_size(WORKGROUP_SIZE)\n"
    << "fn " << entry_name
    << "(@builtin(local_invocation_id) lid: vec3u,\n"
    << " @builtin(workgroup_id) wg_id: vec3u) {\n"
    << "  let tid = lid.x;\n"
    << "  let wg = wg_id.x;\n"
    << "  let H = params.H;\n"
    << "  let H_kv = params.H_kv;\n"
    << "  let L = params.L;\n"
    << "  let gqa = params.gqa_factor;\n"
    << "  let scale = params.scale;\n"
    << "  let b = wg / H;\n"
    << "  let h = wg % H;\n"
    << "  let h_kv = h / gqa;\n"
    << "\n"
    << "  // Offsets (element-indexed) for this (b, h) slice:\n"
    << "  //   Q[b, h, 0, :]   = q[b * H  * D + h    * D + ...]     (Tq=1)\n"
    << "  //   K[b, hk, l, :]  = k[(b*H_kv + hk) * L * D + l*D + d]\n"
    << "  //   V[b, hk, l, :]  = v[(b*H_kv + hk) * L * D + l*D + d]\n"
    << "  //   O[b, h, 0, :]   = o[(b*H + h) * D + d]\n"
    << "  let q_base = (b * H + h) * D;\n"
    << "  let kv_base = (b * H_kv + h_kv) * L * D;\n"
    << "  let o_base = (b * H + h) * D;\n";
  if (has_mask) {
    // Mask is broadcast to output query shape [B, H, 1, L]; row-contiguous.
    s << "  let mask_base = (b * H + h) * L;\n";
  }
  s << "\n";

  // Phase 0: cache Q into shared memory.
  s << "  // Phase 0: cooperatively cache Q[b, h, 0, :] into shared memory.\n"
    << "  if (tid < D) {\n"
    << "    shared_q[tid] = f32(q[q_base + tid]);\n"
    << "  }\n"
    << "  workgroupBarrier();\n"
    << "\n";

  // Phase A-1: compute max of scores over L.
  s << "  // Phase A-1: compute per-thread partial max over L (stride WG).\n"
    << "  var thread_max: f32 = f32(-3.402823e+38);\n"
    << "  for (var l: u32 = tid; l < L; l = l + WORKGROUP_SIZE) {\n"
    << "    var acc: f32 = 0.0;\n"
    << "    let k_row = kv_base + l * D;\n"
    << "    for (var d: u32 = 0u; d < D; d = d + 1u) {\n"
    << "      acc = acc + shared_q[d] * f32(k[k_row + d]);\n"
    << "    }\n"
    << "    var score = acc * scale;\n";
  if (has_mask) {
    s << "    score = score + f32(mask[mask_base + l]);\n";
  }
  s << "    thread_max = max(thread_max, score);\n"
    << "  }\n"
    << "\n";

  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s,
        "thread_max",
        "shared_red",
        "subgroupMax",
        "max_op",
        SDPA_WORKGROUP_SIZE,
        32,
        "mx");
  } else {
    s << "  shared_red[tid] = thread_max;\n"
      << "  workgroupBarrier();\n";
    wgpu::emit_unrolled_reduction(
        s, "shared_red", "max_op", SDPA_WORKGROUP_SIZE);
  }

  // Broadcast the row max via shared memory.
  s << "  if (tid == 0u) { shared_m = shared_red[0]; }\n"
    << "  workgroupBarrier();\n"
    << "  let row_max = shared_m;\n"
    << "\n";

  // Phase A-2: compute sum of exp(score - max) over L.
  s << "  // Phase A-2: compute per-thread partial sum of exp(score - max).\n"
    << "  var thread_sum: f32 = 0.0;\n"
    << "  for (var l: u32 = tid; l < L; l = l + WORKGROUP_SIZE) {\n"
    << "    var acc: f32 = 0.0;\n"
    << "    let k_row = kv_base + l * D;\n"
    << "    for (var d: u32 = 0u; d < D; d = d + 1u) {\n"
    << "      acc = acc + shared_q[d] * f32(k[k_row + d]);\n"
    << "    }\n"
    << "    var score = acc * scale;\n";
  if (has_mask) {
    s << "    score = score + f32(mask[mask_base + l]);\n";
  }
  s << "    thread_sum = thread_sum + exp(score - row_max);\n"
    << "  }\n"
    << "\n";

  if (use_subgroups) {
    wgpu::emit_subgroup_reduction(
        s,
        "thread_sum",
        "shared_red",
        "subgroupAdd",
        "sum_op",
        SDPA_WORKGROUP_SIZE,
        32,
        "sm");
  } else {
    s << "  shared_red[tid] = thread_sum;\n"
      << "  workgroupBarrier();\n";
    wgpu::emit_unrolled_reduction(
        s, "shared_red", "sum_op", SDPA_WORKGROUP_SIZE);
  }

  // Broadcast 1/sum via shared memory.
  s << "  if (tid == 0u) { shared_inv_s = 1.0 / shared_red[0]; }\n"
    << "  workgroupBarrier();\n"
    << "  let inv_s = shared_inv_s;\n"
    << "\n";

  // Phase B: each thread d = tid (if tid < D) accumulates one output lane.
  s << "  // Phase B: each thread owns output lane d = tid.\n"
    << "  // Recompute scores + weighted V accumulation over L.\n"
    << "  if (tid < D) {\n"
    << "    var o_acc: f32 = 0.0;\n"
    << "    for (var l: u32 = 0u; l < L; l = l + 1u) {\n"
    << "      // Recompute score (could be cached, but L >> WG typically).\n"
    << "      var acc: f32 = 0.0;\n"
    << "      let k_row = kv_base + l * D;\n"
    << "      for (var d: u32 = 0u; d < D; d = d + 1u) {\n"
    << "        acc = acc + shared_q[d] * f32(k[k_row + d]);\n"
    << "      }\n"
    << "      var score = acc * scale;\n";
  if (has_mask) {
    s << "      score = score + f32(mask[mask_base + l]);\n";
  }
  s << "      let w = exp(score - row_max) * inv_s;\n"
    << "      let v_row = kv_base + l * D;\n"
    << "      o_acc = o_acc + w * f32(v[v_row + tid]);\n"
    << "    }\n"
    << "    output[o_base + tid] = " << dtype << "(o_acc);\n"
    << "  }\n"
    << "}\n";

  return s.str();
}

} // namespace

// ---------------------------------------------------------------------------
// ScaledDotProductAttention::use_fallback
// ---------------------------------------------------------------------------

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  // Training / logsumexp path uses the VJP-friendly decomposed kernel.
  if (is_training || output_logsumexp) {
    return true;
  }
  // Sanity: Q/K/V must have at least [B, H, T, D] layout.
  if (q.ndim() < 2 || k.ndim() < 2 || v.ndim() < 2) {
    return true;
  }

  // VECTOR path only: Tq == 1. (do_causal is a no-op when Tq == 1, accepted.)
  int tq = q.shape(-2);
  if (tq != 1) {
    return true;
  }

  // Head dim restriction: D in {64, 96, 128}. Gives us unrolling-friendly
  // shapes and matches the 128-thread workgroup size (one thread per lane).
  int d = q.shape(-1);
  if (d != 64 && d != 96 && d != 128) {
    return true;
  }
  if (k.shape(-1) != d || v.shape(-1) != d) {
    return true;
  }

  // K and V must have matching sequence length.
  if (k.shape(-2) != v.shape(-2)) {
    return true;
  }

  // Dtype restriction: float32 or bfloat16 (which is promoted to f32 on GPU).
  auto dt = q.dtype();
  if (dt != float32 && dt != bfloat16) {
    return true;
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    return true;
  }

  // Mask handling:
  //   - No mask -> OK.
  //   - Array mask (non-bool) -> OK (we read it as additive float).
  //   - Array mask (bool) -> NOT OK: we say supports_bool_mask() == false,
  //     so fast.cpp will convert it to an additive float mask before reaching
  //     eval_gpu. We still accept has_arr_mask here.
  //   - Scalar bool mask (has_mask && !has_arr_mask) -> fall back.
  if (has_mask && !has_arr_mask) {
    return true;
  }

  // Row-contiguity: the kernel indexes Q/K/V as flat arrays. Non-contig inputs
  // would need per-dim stride arithmetic which we don't implement yet.
  if (!q.flags().row_contiguous || !k.flags().row_contiguous ||
      !v.flags().row_contiguous) {
    return true;
  }

  // NOTE on mask row-contiguity: fast.cpp broadcasts the incoming mask to
  // shape [B, H, 1, L], which often produces strided (broadcast) arrays.
  // The kernel assumes row-contiguous mask layout for the [B, H, 1, L]
  // shape and indexes it as `mask[(b*H + h)*L + l]`. We cannot check the
  // mask array here (it's built after use_fallback runs), so eval_gpu makes
  // the mask contiguous via contiguous_copy_gpu if needed.

  return false;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

// ---------------------------------------------------------------------------
// ScaledDotProductAttention::eval_gpu
// ---------------------------------------------------------------------------

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Inputs layout from fast.cpp:
  //   inputs[0] = Q
  //   inputs[1] = K
  //   inputs[2] = V
  //   inputs[3] = mask (if has_arr_mask; always present when we accept a mask)
  //   inputs[back] = sinks (we do not support sinks; use_fallback rejects that
  //                         path earlier if the sinks flag is set, but defend
  //                         here anyway).
  assert(outputs.size() == 1);
  auto& out = outputs[0];
  auto& s = stream();

  if (has_sinks_) {
    // The vector kernel does not support attention sinks. use_fallback is a
    // static method and does not receive the has_sinks flag, so we can't
    // reject sinks earlier. For Qwen3.5 inference (the only consumer of this
    // path today) sinks are never set, so this throw is strictly defensive.
    throw std::runtime_error(
        "[WebGPU sdpa] attention sinks are not supported by the fused "
        "vector kernel.");
  }

  // Figure out whether we have an array mask. Mirrors the Metal backend's
  // convention: has_arr_mask when inputs has more than (3 + has_sinks_)
  // entries. Since we rejected has_sinks_ above, this reduces to
  // inputs.size() > 3.
  bool has_mask = inputs.size() > 3;
  if (inputs.size() != 3 && inputs.size() != 4) {
    throw std::runtime_error(
        "[WebGPU sdpa] Unexpected input count " +
        std::to_string(inputs.size()) + "; expected 3 or 4.");
  }

  const array& q = inputs[0];
  const array& k = inputs[1];
  const array& v = inputs[2];

  // Shapes: Q = [B, H, 1, D], K/V = [B, H_kv, L, D].
  uint32_t B = static_cast<uint32_t>(q.shape(0));
  uint32_t H = static_cast<uint32_t>(q.shape(1));
  uint32_t H_kv = static_cast<uint32_t>(k.shape(1));
  uint32_t L = static_cast<uint32_t>(k.shape(-2));
  uint32_t D = static_cast<uint32_t>(q.shape(-1));

  if (H_kv == 0 || H % H_kv != 0) {
    throw std::runtime_error(
        std::string("[WebGPU sdpa] invalid head counts H=") +
        std::to_string(H) + " H_kv=" + std::to_string(H_kv));
  }
  uint32_t gqa_factor = H / H_kv;

  // Ensure Q/K/V have the right row-contiguous, zero-offset layout. If the
  // upstream arrays are offset (e.g. due to slicing) we fall back to a
  // defensive contiguous copy. use_fallback already rejected non-contig
  // layouts, but offset != 0 can still slip through.
  auto ensure_contig = [&](const array& a) -> array {
    if (a.flags().row_contiguous && a.offset() == 0) {
      return a;
    }
    array c = contiguous_copy_gpu(a, s);
    auto& enc = wgpu::get_command_encoder(s);
    enc.add_temporary(c);
    return c;
  };
  array q_c = ensure_contig(q);
  array k_c = ensure_contig(k);
  array v_c = ensure_contig(v);
  // The mask, if present, is usually a broadcast view of a smaller mask
  // (e.g. [1, 1, 1, L] broadcast to [B, H, 1, L]); make it contiguous so
  // the kernel can read it with a flat offset.
  std::optional<array> mask_c_opt;
  if (has_mask) {
    mask_c_opt = ensure_contig(inputs[3]);
  }

  // Allocate the output.
  out.set_data(allocator::malloc(wgpu::wgpu_alloc_size(out)));
  wgpu::ensure_wgpu_size(out);

  if (B == 0 || H == 0 || D == 0) {
    return;
  }
  // L == 0 is undefined for softmax(Q*K^T); zero the output and return.
  if (L == 0) {
    // Leave output as uninitialised allocation — MLX upstream should never
    // hand us L == 0 for a real attention call. Do nothing; safer than
    // dispatching over an empty range.
    return;
  }

  auto& dev = wgpu::device();
  auto& encoder = wgpu::get_command_encoder(s);

  // Choose WGSL dtype. bf16 is uploaded as f32 by device.cpp, so
  // dtype_to_wgsl_safe returns "f32" for bf16 already.
  const char* dtype_wgsl = wgpu::dtype_to_wgsl_safe(q.dtype());
  std::string dtype(dtype_wgsl);

  bool use_sg = dev.has_subgroups();
  std::string sg_suffix = use_sg ? "_sg" : "";
  std::string mask_suffix = has_mask ? "_m" : "";
  std::string entry_name = std::string("sdpa_v_") + dtype + "_D" +
      std::to_string(D) + mask_suffix + sg_suffix;

  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name, [&]() {
        return make_sdpa_vector_kernel(
            entry_name, dtype, D, has_mask, use_sg);
      });
  auto pe =
      dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  // Register inputs/outputs with the encoder for bookkeeping.
  encoder.set_input_array(q_c);
  encoder.set_input_array(k_c);
  encoder.set_input_array(v_c);
  if (has_mask) {
    encoder.set_input_array(*mask_c_opt);
  }
  encoder.set_output_array(out);

  // Populate uniform buffer.
  SdpaParams params{};
  params.B = B;
  params.H = H;
  params.H_kv = H_kv;
  params.L = L;
  params.D = D;
  params.has_mask = has_mask ? 1u : 0u;
  params.gqa_factor = gqa_factor;
  params.scale = scale_;

  auto& pool = dev.uniform_pool();
  WGPUBuffer uniform_buf =
      pool.acquire(dev.gpu_queue(), &params, sizeof(SdpaParams));

  // Build bind group.
  WGPUBuffer q_buf = wgpu::wgpu_buffer(q_c);
  WGPUBuffer k_buf = wgpu::wgpu_buffer(k_c);
  WGPUBuffer v_buf = wgpu::wgpu_buffer(v_c);
  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);

  std::vector<std::pair<WGPUBuffer, uint64_t>> bg_entries;
  bg_entries.reserve(has_mask ? 6 : 5);
  bg_entries.emplace_back(q_buf, wgpuBufferGetSize(q_buf));
  bg_entries.emplace_back(k_buf, wgpuBufferGetSize(k_buf));
  bg_entries.emplace_back(v_buf, wgpuBufferGetSize(v_buf));
  if (has_mask) {
    WGPUBuffer mask_buf = wgpu::wgpu_buffer(*mask_c_opt);
    bg_entries.emplace_back(mask_buf, wgpuBufferGetSize(mask_buf));
  }
  bg_entries.emplace_back(out_buf, wgpuBufferGetSize(out_buf));
  bg_entries.emplace_back(uniform_buf, sizeof(SdpaParams));

  WGPUBindGroup bg = wgpu::create_bind_group(pe.layout, bg_entries);

  // One workgroup per (batch, head).
  encoder.dispatch_compute(pe.pipeline, bg, B * H, 1u, 1u);

  wgpuBindGroupRelease(bg);
  encoder.add_completed_handler([uniform_buf]() {
    wgpu::device().uniform_pool().release(uniform_buf);
  });
}

} // namespace mlx::core::fast
