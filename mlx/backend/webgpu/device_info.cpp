// Copyright 2026 Apple Inc.

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/webgpu/device.h"
#include "mlx/backend/webgpu/webgpu_backend.h"

namespace mlx::core::gpu {

bool is_available() {
  try {
    return wgpu::device().is_valid();
  } catch (const std::runtime_error&) {
    return false;
  }
}

int device_count() {
  return is_available() ? 1 : 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  static auto info = []() {
    std::unordered_map<std::string, std::variant<std::string, size_t>> result;

    auto& dev = wgpu::device();
    if (!dev.is_valid()) {
      return result;
    }

    // Query adapter info - try the newer wgpuAdapterGetInfo first,
    // fall back to wgpuAdapterGetProperties.
    // Dawn uses WGPUAdapterProperties with wgpuAdapterGetProperties.
    WGPUAdapterProperties props = {};
    wgpuAdapterGetProperties(dev.gpu_adapter(), &props);

    result["device_name"] =
        std::string(props.name ? props.name : "WebGPU Device");
    result["architecture"] =
        std::string(props.architecture ? props.architecture : "Unknown");
    if (props.driverDescription) {
      result["driver_description"] = std::string(props.driverDescription);
    }

    // Map backend type to string
    const char* backend_str = "Unknown";
    switch (props.backendType) {
      case WGPUBackendType_Vulkan:
        backend_str = "Vulkan";
        break;
      case WGPUBackendType_Metal:
        backend_str = "Metal";
        break;
      case WGPUBackendType_D3D12:
        backend_str = "D3D12";
        break;
      case WGPUBackendType_D3D11:
        backend_str = "D3D11";
        break;
      case WGPUBackendType_OpenGL:
        backend_str = "OpenGL";
        break;
      case WGPUBackendType_OpenGLES:
        backend_str = "OpenGLES";
        break;
      default:
        break;
    }
    result["backend_type"] = std::string(backend_str);

    // Query device limits for info
    WGPUSupportedLimits limits = {};
    if (wgpuDeviceGetLimits(dev.gpu_device(), &limits)) {
      result["max_buffer_size"] =
          static_cast<size_t>(limits.limits.maxBufferSize);
      result["max_storage_buffer_binding_size"] =
          static_cast<size_t>(limits.limits.maxStorageBufferBindingSize);
    }

    return result;
  }();
  return info;
}

} // namespace mlx::core::gpu

namespace mlx::core::wgpu {

bool is_available() {
  return gpu::is_available();
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  return gpu::device_info(device_index);
}

} // namespace mlx::core::wgpu
