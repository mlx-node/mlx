# WebGPU backend for MLX
#
# Uses the standard webgpu.h C API header.
# Selectable implementations: Dawn or wgpu-native via WEBGPU_BACKEND variable.

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_WEBGPU)

# ---------- WebGPU implementation ----------
# Use WEBGPU_BACKEND to select: DAWN or WGPU (default: DAWN)
if(NOT DEFINED WEBGPU_BACKEND)
  set(WEBGPU_BACKEND "DAWN" CACHE STRING "WebGPU implementation (DAWN or WGPU)")
endif()

if(WEBGPU_BACKEND STREQUAL "DAWN")
  # Fetch Dawn from Google's repo
  include(FetchContent)
  FetchContent_Declare(
    dawn
    GIT_REPOSITORY https://dawn.googlesource.com/dawn
    GIT_TAG main
    GIT_SHALLOW TRUE)

  # Dawn build options
  set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
  set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  set(DAWN_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(dawn)

  target_link_libraries(mlx PRIVATE webgpu_dawn webgpu_cpp)
  target_include_directories(mlx PRIVATE ${dawn_SOURCE_DIR}/include)
  target_compile_definitions(mlx PRIVATE WEBGPU_BACKEND_DAWN)

elseif(WEBGPU_BACKEND STREQUAL "WGPU")
  # Fetch wgpu-native pre-built release
  include(FetchContent)

  # Determine platform-specific archive
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(WGPU_PLATFORM "linux-x86_64")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(WGPU_PLATFORM "macos-aarch64")
    else()
      set(WGPU_PLATFORM "macos-x86_64")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(WGPU_PLATFORM "windows-x86_64-msvc")
  endif()

  set(WGPU_VERSION "v24.0.0.2")
  FetchContent_Declare(
    wgpu_native
    URL "https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_VERSION}/wgpu-${WGPU_PLATFORM}-release.zip")
  FetchContent_MakeAvailable(wgpu_native)

  target_include_directories(mlx PRIVATE ${wgpu_native_SOURCE_DIR}/include)
  find_library(WGPU_LIB wgpu_native PATHS ${wgpu_native_SOURCE_DIR}/lib)
  target_link_libraries(mlx PRIVATE ${WGPU_LIB})
  target_compile_definitions(mlx PRIVATE WEBGPU_BACKEND_WGPU)

elseif(WEBGPU_BACKEND STREQUAL "WASI_IMPORT")
  # WASI/WASM path: WebGPU functions are provided as WASM imports from JS bridge.
  # No library to link — functions resolve at WASM instantiation time.
  # Use the bundled minimal webgpu.h header.
  message(STATUS "WebGPU backend: WASI_IMPORT (functions provided by JS bridge)")
  target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

else()
  message(FATAL_ERROR "Unknown WEBGPU_BACKEND: ${WEBGPU_BACKEND}. Use DAWN, WGPU, or WASI_IMPORT.")
endif()

# ---------- Embed WGSL kernel sources ----------
# Convert .wgsl files to C string constants (similar to CUDA's bin2h.cmake)
file(GLOB WGSL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.wgsl")
set(WGSL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/gen/wgsl_sources.h")
set(WGSL_HEADER_CONTENT "#pragma once\n// Auto-generated WGSL kernel sources\nnamespace mlx::core::wgpu::kernels {\n")

foreach(WGSL_FILE ${WGSL_SOURCES})
  get_filename_component(WGSL_NAME ${WGSL_FILE} NAME_WE)
  file(READ ${WGSL_FILE} WGSL_CONTENT)
  string(REPLACE "\\" "\\\\" WGSL_CONTENT "${WGSL_CONTENT}")
  string(REPLACE "\"" "\\\"" WGSL_CONTENT "${WGSL_CONTENT}")
  string(REPLACE "\n" "\\n\"\n\"" WGSL_CONTENT "${WGSL_CONTENT}")
  string(APPEND WGSL_HEADER_CONTENT "inline const char* ${WGSL_NAME}() {\n  return \"${WGSL_CONTENT}\";\n}\n\n")
endforeach()

string(APPEND WGSL_HEADER_CONTENT "} // namespace mlx::core::wgpu::kernels\n")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(WRITE ${WGSL_HEADER} "${WGSL_HEADER_CONTENT}")
target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

