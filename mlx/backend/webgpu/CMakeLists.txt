# WebGPU backend for MLX
#
# Uses the standard webgpu.h C API header.
# Selectable implementations: Dawn or wgpu-native via WEBGPU_BACKEND variable.
#
# CG-F014: this file is only included from the parent mlx/CMakeLists.txt
# when MLX_BUILD_WEBGPU is ON (see the if/else block there — the OFF arm
# compiles just no_webgpu.cpp). The defensive guard below catches the
# case where someone add_subdirectory()'s this file directly without
# going through the parent gate, so a mis-wired project surfaces an
# actionable error instead of quietly linking the full backend.
if(DEFINED MLX_BUILD_WEBGPU AND NOT MLX_BUILD_WEBGPU)
  message(FATAL_ERROR
    "backend/webgpu/CMakeLists.txt was included with MLX_BUILD_WEBGPU=OFF. "
    "The parent CMakeLists should be compiling no_webgpu.cpp instead. "
    "Either enable MLX_BUILD_WEBGPU or remove the add_subdirectory() call.")
endif()

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
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/normalization.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sdpa.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_WEBGPU)

# DEV-F023: WEBGPU_PROFILE gates the hot-path observability atomics in
# device.cpp (g_total_dispatches / g_total_pass_ends / op_count_kernels_).
# Default ON for dev builds; benchmark builds pass -DWEBGPU_PROFILE=0
# on the CMake command line. See the header of device.cpp for details.
if(NOT DEFINED WEBGPU_PROFILE)
  set(WEBGPU_PROFILE 1 CACHE STRING "Enable WebGPU per-dispatch profile counters")
endif()
target_compile_definitions(mlx PRIVATE WEBGPU_PROFILE=${WEBGPU_PROFILE})

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
  # CG-F016: NO_DEFAULT_PATH locks the search to the FetchContent-extracted
  # tarball so a system-installed wgpu_native (a different version, maybe
  # ABI-incompatible) cannot shadow the pinned WGPU_VERSION above. REQUIRED
  # fails configuration cleanly if the tarball didn't contain the library.
  find_library(WGPU_LIB
    NAMES wgpu_native
    PATHS ${wgpu_native_SOURCE_DIR}/lib
    NO_DEFAULT_PATH
    REQUIRED)
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
# CG-F015: previously this ran as file(READ/WRITE) at CMake configure time,
# so editing a .wgsl file did NOT retrigger the embed — you had to rerun
# cmake. Now the header is generated via add_custom_command driven by a
# standalone CMake script, with DEPENDS on every .wgsl source and the
# script itself. Builds pick up .wgsl edits without re-configuring.
#
# CONFIGURE_DEPENDS lets the glob pick up added/removed .wgsl files at
# build-tree regen; pairs with the custom-command DEPENDS for per-file
# edits.
file(GLOB WGSL_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.wgsl")
set(WGSL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/gen/wgsl_sources.h")
set(WGSL_EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/gen/embed_wgsl.cmake")

# Write a script that does the same READ/escape/WRITE the old code did,
# but invoked at build time by cmake -P. Bracket argument [=[...]=] keeps
# the embedded CMake code verbatim (no ${...} expansion here), so the
# escape sequences inside the generated script survive the outer quoting.
file(WRITE ${WGSL_EMBED_SCRIPT} [=[
# Auto-generated by backend/webgpu/CMakeLists.txt - do not edit.
# Usage: cmake -DWGSL_SOURCES="a.wgsl;b.wgsl" -DWGSL_HEADER=out.h -P embed_wgsl.cmake
set(CONTENT "#pragma once\n// Auto-generated WGSL kernel sources\nnamespace mlx::core::wgpu::kernels {\n")
foreach(WGSL_FILE ${WGSL_SOURCES})
  get_filename_component(WGSL_NAME ${WGSL_FILE} NAME_WE)
  file(READ ${WGSL_FILE} WGSL_CONTENT)
  string(REPLACE "\\" "\\\\" WGSL_CONTENT "${WGSL_CONTENT}")
  string(REPLACE "\"" "\\\"" WGSL_CONTENT "${WGSL_CONTENT}")
  string(REPLACE "\n" "\\n\"\n\"" WGSL_CONTENT "${WGSL_CONTENT}")
  string(APPEND CONTENT "inline const char* ${WGSL_NAME}() {\n  return \"${WGSL_CONTENT}\";\n}\n\n")
endforeach()
string(APPEND CONTENT "} // namespace mlx::core::wgpu::kernels\n")
get_filename_component(OUT_DIR ${WGSL_HEADER} DIRECTORY)
file(MAKE_DIRECTORY ${OUT_DIR})
file(WRITE ${WGSL_HEADER} "${CONTENT}")
]=])

add_custom_command(
  OUTPUT ${WGSL_HEADER}
  COMMAND ${CMAKE_COMMAND}
          -DWGSL_SOURCES=${WGSL_SOURCES}
          -DWGSL_HEADER=${WGSL_HEADER}
          -P ${WGSL_EMBED_SCRIPT}
  DEPENDS ${WGSL_SOURCES} ${WGSL_EMBED_SCRIPT}
  COMMENT "Embedding WGSL kernel sources -> wgsl_sources.h"
  VERBATIM)

add_custom_target(mlx_wgsl_embed DEPENDS ${WGSL_HEADER})
add_dependencies(mlx mlx_wgsl_embed)

target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

