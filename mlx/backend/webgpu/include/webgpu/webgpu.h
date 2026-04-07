/**
 * Minimal webgpu.h for WASI-SDK builds.
 *
 * Declares only the types and functions used by MLX's WebGPU backend,
 * with Dawn-style callback signatures (callback + userdata as separate args).
 *
 * All functions become unresolved WASM imports, satisfied at runtime by the
 * JavaScript WebGPU bridge.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------- Opaque handle types ----------
typedef struct WGPUAdapterImpl* WGPUAdapter;
typedef struct WGPUBindGroupImpl* WGPUBindGroup;
typedef struct WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;
typedef struct WGPUBufferImpl* WGPUBuffer;
typedef struct WGPUCommandBufferImpl* WGPUCommandBuffer;
typedef struct WGPUCommandEncoderImpl* WGPUCommandEncoder;
typedef struct WGPUComputePassEncoderImpl* WGPUComputePassEncoder;
typedef struct WGPUComputePipelineImpl* WGPUComputePipeline;
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUInstanceImpl* WGPUInstance;
typedef struct WGPUPipelineLayoutImpl* WGPUPipelineLayout;
typedef struct WGPUQueueImpl* WGPUQueue;
typedef struct WGPUShaderModuleImpl* WGPUShaderModule;

// ---------- Basic types ----------
typedef uint64_t WGPUFlags;
typedef WGPUFlags WGPUBufferUsageFlags;
typedef WGPUFlags WGPUBufferUsage;
typedef uint32_t WGPUBool;
typedef uint32_t WGPUMapModeFlags;

// ---------- Enums ----------
typedef enum WGPURequestAdapterStatus {
    WGPURequestAdapterStatus_Success = 0,
    WGPURequestAdapterStatus_Unavailable = 1,
    WGPURequestAdapterStatus_Error = 2,
} WGPURequestAdapterStatus;

typedef enum WGPURequestDeviceStatus {
    WGPURequestDeviceStatus_Success = 0,
    WGPURequestDeviceStatus_Error = 1,
} WGPURequestDeviceStatus;

typedef enum WGPUBufferMapAsyncStatus {
    WGPUBufferMapAsyncStatus_Success = 0,
    WGPUBufferMapAsyncStatus_Error = 1,
} WGPUBufferMapAsyncStatus;

typedef enum WGPUQueueWorkDoneStatus {
    WGPUQueueWorkDoneStatus_Success = 0,
    WGPUQueueWorkDoneStatus_Error = 1,
} WGPUQueueWorkDoneStatus;

typedef enum WGPUErrorType {
    WGPUErrorType_NoError = 0,
    WGPUErrorType_Validation = 1,
    WGPUErrorType_OutOfMemory = 2,
    WGPUErrorType_Internal = 3,
    WGPUErrorType_Unknown = 4,
    WGPUErrorType_DeviceLost = 5,
} WGPUErrorType;

typedef enum WGPUDeviceLostReason {
    WGPUDeviceLostReason_Undefined = 0,
    WGPUDeviceLostReason_Destroyed = 1,
} WGPUDeviceLostReason;

typedef enum WGPUBufferMapState {
    WGPUBufferMapState_Unmapped = 0,
    WGPUBufferMapState_Pending = 1,
    WGPUBufferMapState_Mapped = 2,
} WGPUBufferMapState;

typedef enum WGPUFeatureName {
    WGPUFeatureName_Undefined = 0,
    WGPUFeatureName_ShaderF16 = 14,
} WGPUFeatureName;

typedef enum WGPUPowerPreference {
    WGPUPowerPreference_Undefined = 0,
    WGPUPowerPreference_LowPower = 1,
    WGPUPowerPreference_HighPerformance = 2,
} WGPUPowerPreference;

typedef enum WGPUAdapterType {
    WGPUAdapterType_DiscreteGPU = 0,
    WGPUAdapterType_IntegratedGPU = 1,
    WGPUAdapterType_CPU = 2,
    WGPUAdapterType_Unknown = 3,
} WGPUAdapterType;

typedef enum WGPUBackendType {
    WGPUBackendType_Undefined = 0,
    WGPUBackendType_Null = 1,
    WGPUBackendType_WebGPU = 2,
    WGPUBackendType_D3D11 = 3,
    WGPUBackendType_D3D12 = 4,
    WGPUBackendType_Metal = 5,
    WGPUBackendType_Vulkan = 6,
    WGPUBackendType_OpenGL = 7,
    WGPUBackendType_OpenGLES = 8,
} WGPUBackendType;

typedef enum WGPUSType {
    WGPUSType_ShaderSourceWGSL = 5,
    WGPUSType_ShaderModuleWGSLDescriptor = 5, // Dawn compat alias
} WGPUSType;

// ---------- Buffer usage flags ----------
static const WGPUBufferUsageFlags WGPUBufferUsage_None = 0x00000000;
static const WGPUBufferUsageFlags WGPUBufferUsage_MapRead = 0x00000001;
static const WGPUBufferUsageFlags WGPUBufferUsage_MapWrite = 0x00000002;
static const WGPUBufferUsageFlags WGPUBufferUsage_CopySrc = 0x00000004;
static const WGPUBufferUsageFlags WGPUBufferUsage_CopyDst = 0x00000008;
static const WGPUBufferUsageFlags WGPUBufferUsage_Index = 0x00000010;
static const WGPUBufferUsageFlags WGPUBufferUsage_Vertex = 0x00000020;
static const WGPUBufferUsageFlags WGPUBufferUsage_Uniform = 0x00000040;
static const WGPUBufferUsageFlags WGPUBufferUsage_Storage = 0x00000080;
static const WGPUBufferUsageFlags WGPUBufferUsage_Indirect = 0x00000100;
static const WGPUBufferUsageFlags WGPUBufferUsage_QueryResolve = 0x00000200;

// Map mode flags
static const WGPUMapModeFlags WGPUMapMode_None = 0x00000000;
static const WGPUMapModeFlags WGPUMapMode_Read = 0x00000001;
static const WGPUMapModeFlags WGPUMapMode_Write = 0x00000002;

// ---------- Chained struct base ----------
typedef struct WGPUChainedStruct {
    struct WGPUChainedStruct const* next;
    WGPUSType sType;
} WGPUChainedStruct;

// ---------- Descriptors ----------
typedef struct WGPUInstanceDescriptor {
    WGPUChainedStruct const* nextInChain;
} WGPUInstanceDescriptor;

typedef struct WGPURequestAdapterOptions {
    WGPUChainedStruct const* nextInChain;
    WGPUPowerPreference powerPreference;
} WGPURequestAdapterOptions;

typedef struct WGPULimits {
    uint32_t maxTextureDimension1D;
    uint32_t maxTextureDimension2D;
    uint32_t maxTextureDimension3D;
    uint32_t maxTextureArrayLayers;
    uint32_t maxBindGroups;
    uint32_t maxBindGroupsPlusVertexBuffers;
    uint32_t maxBindingsPerBindGroup;
    uint32_t maxDynamicUniformBuffersPerPipelineLayout;
    uint32_t maxDynamicStorageBuffersPerPipelineLayout;
    uint32_t maxSampledTexturesPerShaderStage;
    uint32_t maxSamplersPerShaderStage;
    uint32_t maxStorageBuffersPerShaderStage;
    uint32_t maxStorageTexturesPerShaderStage;
    uint32_t maxUniformBuffersPerShaderStage;
    uint64_t maxUniformBufferBindingSize;
    uint64_t maxStorageBufferBindingSize;
    uint32_t minUniformBufferOffsetAlignment;
    uint32_t minStorageBufferOffsetAlignment;
    uint32_t maxVertexBuffers;
    uint64_t maxBufferSize;
    uint32_t maxVertexAttributes;
    uint32_t maxVertexBufferArrayStride;
    uint32_t maxInterStageShaderComponents;
    uint32_t maxInterStageShaderVariables;
    uint32_t maxColorAttachments;
    uint32_t maxColorAttachmentBytesPerSample;
    uint32_t maxComputeWorkgroupStorageSize;
    uint32_t maxComputeInvocationsPerWorkgroup;
    uint32_t maxComputeWorkgroupSizeX;
    uint32_t maxComputeWorkgroupSizeY;
    uint32_t maxComputeWorkgroupSizeZ;
    uint32_t maxComputeWorkgroupsPerDimension;
} WGPULimits;

typedef struct WGPURequiredLimits {
    WGPUChainedStruct const* nextInChain;
    WGPULimits limits;
} WGPURequiredLimits;

typedef struct WGPUSupportedLimits {
    WGPUChainedStruct* nextInChain;
    WGPULimits limits;
} WGPUSupportedLimits;

typedef struct WGPUQueueDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
} WGPUQueueDescriptor;

typedef struct WGPUDeviceDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
    size_t requiredFeatureCount;
    WGPUFeatureName const* requiredFeatures;
    WGPURequiredLimits const* requiredLimits;
    WGPUQueueDescriptor defaultQueue; // Dawn compat
} WGPUDeviceDescriptor;

typedef struct WGPUBufferDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
    WGPUBufferUsageFlags usage;
    uint64_t size;
    WGPUBool mappedAtCreation;
} WGPUBufferDescriptor;

typedef struct WGPUShaderModuleWGSLDescriptor {
    WGPUChainedStruct chain;
    char const* code;
} WGPUShaderModuleWGSLDescriptor;

typedef struct WGPUShaderModuleDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
} WGPUShaderModuleDescriptor;

typedef struct WGPUBindGroupEntry {
    WGPUChainedStruct const* nextInChain;
    uint32_t binding;
    WGPUBuffer buffer;
    uint64_t offset;
    uint64_t size;
    void* sampler;
    void* textureView;
} WGPUBindGroupEntry;

typedef struct WGPUBindGroupDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
    WGPUBindGroupLayout layout;
    size_t entryCount;
    WGPUBindGroupEntry const* entries;
} WGPUBindGroupDescriptor;

typedef struct WGPUProgrammableStageDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUShaderModule module;
    char const* entryPoint;
    size_t constantCount;
    void const* constants;
} WGPUProgrammableStageDescriptor;

typedef struct WGPUComputePipelineDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
    WGPUPipelineLayout layout;
    WGPUProgrammableStageDescriptor compute;
} WGPUComputePipelineDescriptor;

typedef struct WGPUCommandEncoderDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
} WGPUCommandEncoderDescriptor;

typedef struct WGPUComputePassDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
} WGPUComputePassDescriptor;

typedef struct WGPUCommandBufferDescriptor {
    WGPUChainedStruct const* nextInChain;
    char const* label;
} WGPUCommandBufferDescriptor;

typedef struct WGPUAdapterProperties {
    WGPUChainedStruct* nextInChain;
    uint32_t vendorID;
    char const* vendorName;
    char const* architecture;
    uint32_t deviceID;
    char const* name;
    char const* driverDescription;
    WGPUAdapterType adapterType;
    WGPUBackendType backendType;
} WGPUAdapterProperties;

// ---------- Callback types (Dawn-style) ----------
typedef void (*WGPUInstanceRequestAdapterCallback)(
    WGPURequestAdapterStatus status, WGPUAdapter adapter,
    char const* message, void* userdata);

typedef void (*WGPUAdapterRequestDeviceCallback)(
    WGPURequestDeviceStatus status, WGPUDevice device,
    char const* message, void* userdata);

typedef void (*WGPUBufferMapCallback)(
    WGPUBufferMapAsyncStatus status, void* userdata);

typedef void (*WGPUQueueWorkDoneCallback)(
    WGPUQueueWorkDoneStatus status, void* userdata);

typedef void (*WGPUErrorCallback)(
    WGPUErrorType type, char const* message, void* userdata);

typedef void (*WGPUDeviceLostCallback)(
    WGPUDeviceLostReason reason, char const* message, void* userdata);

// ---------- Function declarations ----------

// Instance
WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const* descriptor);
void wgpuInstanceRequestAdapter(WGPUInstance instance,
    WGPURequestAdapterOptions const* options,
    WGPUInstanceRequestAdapterCallback callback, void* userdata);
void wgpuInstanceRelease(WGPUInstance instance);

// Adapter
void wgpuAdapterRequestDevice(WGPUAdapter adapter,
    WGPUDeviceDescriptor const* descriptor,
    WGPUAdapterRequestDeviceCallback callback, void* userdata);
void wgpuAdapterGetProperties(WGPUAdapter adapter, WGPUAdapterProperties* properties);
WGPUBool wgpuAdapterGetLimits(WGPUAdapter adapter, WGPUSupportedLimits* limits);
WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature);
void wgpuAdapterRelease(WGPUAdapter adapter);

// Device
WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, WGPUBufferDescriptor const* descriptor);
WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice device, WGPUShaderModuleDescriptor const* descriptor);
WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device, WGPUComputePipelineDescriptor const* descriptor);
WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice device, WGPUBindGroupDescriptor const* descriptor);
WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice device, WGPUCommandEncoderDescriptor const* descriptor);
WGPUQueue wgpuDeviceGetQueue(WGPUDevice device);
WGPUBool wgpuDeviceGetLimits(WGPUDevice device, WGPUSupportedLimits* limits);
void wgpuDeviceSetUncapturedErrorCallback(WGPUDevice device, WGPUErrorCallback callback, void* userdata);
void wgpuDeviceSetDeviceLostCallback(WGPUDevice device, WGPUDeviceLostCallback callback, void* userdata);
void wgpuDeviceRelease(WGPUDevice device);

// Queue
void wgpuQueueSubmit(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const* commands);
void wgpuQueueWriteBuffer(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, void const* data, size_t size);
void wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallback callback, void* userdata);
void wgpuQueueRelease(WGPUQueue queue);

// Command encoder
WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder encoder, WGPUComputePassDescriptor const* descriptor);
void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder encoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size);
WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder encoder, WGPUCommandBufferDescriptor const* descriptor);
void wgpuCommandEncoderRelease(WGPUCommandEncoder encoder);

// Command buffer
void wgpuCommandBufferRelease(WGPUCommandBuffer commandBuffer);

// Compute pass encoder
void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder encoder, WGPUComputePipeline pipeline);
void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder encoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const* dynamicOffsets);
void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder encoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ);
void wgpuComputePassEncoderEnd(WGPUComputePassEncoder encoder);
void wgpuComputePassEncoderRelease(WGPUComputePassEncoder encoder);

// Buffer
uint64_t wgpuBufferGetSize(WGPUBuffer buffer);
void* wgpuBufferGetMappedRange(WGPUBuffer buffer, size_t offset, size_t size);
void const* wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size);
void wgpuBufferUnmap(WGPUBuffer buffer);
void wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapModeFlags mode, size_t offset, size_t size, WGPUBufferMapCallback callback, void* userdata);
void wgpuBufferDestroy(WGPUBuffer buffer);
void wgpuBufferRelease(WGPUBuffer buffer);

// Pipeline
WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline pipeline, uint32_t groupIndex);
void wgpuComputePipelineRelease(WGPUComputePipeline pipeline);

// Bind group / layout / shader release
void wgpuBindGroupRelease(WGPUBindGroup bindGroup);
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout bindGroupLayout);
void wgpuShaderModuleRelease(WGPUShaderModule shaderModule);

#ifdef __cplusplus
}
#endif
