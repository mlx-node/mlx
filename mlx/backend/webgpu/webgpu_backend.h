// Copyright © 2026 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::wgpu {

/* Check if the WebGPU backend is available. */
MLX_API bool is_available();

/* Get information about a WebGPU device. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

} // namespace mlx::core::wgpu
