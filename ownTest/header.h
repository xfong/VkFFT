// Force OpenCL backend
#define VKFFT_BACKEND 3

typedef struct {
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue commandQueue;
        uint64_t device_id;//an id of a device, reported by Vulkan device list
} VkGPU;//an example structure containing Vulkan primitives

#include "vkFFT.h"
