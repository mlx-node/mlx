// Copyright © 2026 Apple Inc.

#include "mlx/backend/webgpu/webgpu_backend.h"

namespace mlx::core {

namespace wgpu {

bool is_available() {
  return false;
}

// CG-F013: webgpu_backend.h exports device_info(int); when MLX_USE_WEBGPU
// is OFF the full backend's definition in device_info.cpp isn't compiled,
// so without this stub the symbol is undefined and anything that touches
// the header (e.g. Python bindings that expose `mlx.core.wgpu.device_info`)
// fails to link. Return a process-local empty map by reference; callers
// must not mutate it.
const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  static const std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;
  return empty;
}

} // namespace wgpu

} // namespace mlx::core
