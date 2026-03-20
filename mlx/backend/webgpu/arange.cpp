// Copyright 2026 Apple Inc.
//
// WebGPU Arange implementation.
//
// Generates a WGSL kernel that fills an output buffer with
// start + step * idx for each element index.

#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace {

// Uniform buffer layout for Arange kernel (float variant).
struct ArangeParamsFloat {
  uint32_t size;
  float start;
  float step;
  uint32_t _pad;
};

// Uniform buffer layout for Arange kernel (integer variant).
struct ArangeParamsInt {
  uint32_t size;
  int32_t start;
  int32_t step;
  uint32_t _pad;
};

// Generate WGSL kernel for arange.
// For float types: out[idx] = out_type(start + step * f32(idx))
// For int types:   out[idx] = out_type(start + step * i32(idx))
std::string make_arange_kernel(
    const std::string& entry_name,
    const std::string& out_type,
    bool is_integer) {
  std::ostringstream s;

  if (out_type == "f16") {
    s << "enable f16;\n\n";
  }

  if (is_integer) {
    s << "struct ArangeParams {\n"
      << "  size: u32,\n"
      << "  start: i32,\n"
      << "  step: i32,\n"
      << "  _pad: u32,\n"
      << "}\n\n";
  } else {
    s << "struct ArangeParams {\n"
      << "  size: u32,\n"
      << "  start: f32,\n"
      << "  step: f32,\n"
      << "  _pad: u32,\n"
      << "}\n\n";
  }

  s << "@group(0) @binding(0) var<storage, read_write> out: array<"
    << out_type << ">;\n"
    << "@group(0) @binding(1) var<uniform> params: ArangeParams;\n\n";

  s << "@compute @workgroup_size(256)\n"
    << "fn " << entry_name
    << "(@builtin(global_invocation_id) gid: vec3u) {\n"
    << "  let idx = gid.x;\n"
    << "  if (idx >= params.size) { return; }\n";

  if (is_integer) {
    s << "  let val = params.start + params.step * i32(idx);\n"
      << "  out[idx] = " << out_type << "(val);\n";
  } else {
    s << "  let val = params.start + params.step * f32(idx);\n"
      << "  out[idx] = " << out_type << "(val);\n";
  }

  s << "}\n";

  return s.str();
}

} // namespace

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();

  if (out.size() == 0) {
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  const char* out_type = wgpu::dtype_to_wgsl(out.dtype());

  // Use integer params for integer output types to avoid float32 precision loss
  bool is_integer = issubdtype(out.dtype(), integer);

  std::string entry_name =
      std::string("arange_") + out_type + (is_integer ? "_int" : "_float");
  auto& dev = wgpu::device();
  WGPUShaderModule shader = dev.get_or_create_shader_module(
      entry_name,
      [&]() { return make_arange_kernel(entry_name, out_type, is_integer); });
  auto pe = dev.get_or_create_pipeline(entry_name, shader, entry_name.c_str());

  auto& encoder = wgpu::get_command_encoder(s);
  encoder.set_output_array(out);

  // Fill params -- use the appropriate type to avoid precision loss
  uint32_t data_size = static_cast<uint32_t>(out.data_size());
  WGPUBuffer uniform_buf;

  if (is_integer) {
    ArangeParamsInt params{};
    params.size = data_size;
    params.start = static_cast<int32_t>(start_);
    params.step = static_cast<int32_t>(step_);
    uniform_buf =
        wgpu::create_uniform_buffer(&params, sizeof(ArangeParamsInt));
  } else {
    ArangeParamsFloat params{};
    params.size = data_size;
    params.start = static_cast<float>(start_);
    params.step = static_cast<float>(step_);
    uniform_buf =
        wgpu::create_uniform_buffer(&params, sizeof(ArangeParamsFloat));
  }

  WGPUBuffer out_buf = wgpu::wgpu_buffer(out);
  uint64_t out_buf_size = wgpuBufferGetSize(out_buf);

  // Both param structs have the same size (16 bytes)
  size_t params_size =
      is_integer ? sizeof(ArangeParamsInt) : sizeof(ArangeParamsFloat);

  WGPUBindGroup bg = wgpu::create_bind_group(
      pe.layout,
      {{out_buf, out_buf_size},
       {uniform_buf, params_size}});

  uint32_t num_workgroups =
      (data_size + wgpu::WORKGROUP_SIZE - 1) / wgpu::WORKGROUP_SIZE;
  encoder.dispatch_compute(pe.pipeline, bg, num_workgroups);

  wgpuBindGroupRelease(bg);
  wgpuBufferDestroy(uniform_buf);
  wgpuBufferRelease(uniform_buf);
}

} // namespace mlx::core
