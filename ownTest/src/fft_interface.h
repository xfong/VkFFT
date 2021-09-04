// Interface to vkFFT with convenice functions
#define VKFFT_BACKEND     3
#define __SIZEOF_HALF__   2
#define __SIZEOF_FLOAT__  4
#define __SIZEOF_DOUBLE__ 8
#include "vkFFT.h"

// Use a plan structure
struct interfaceFFTPlan {
    VkFFTConfiguration*   config;
    VkFFTApplication*     app;
    bool                  isBaked;
    VkFFTLaunchParams*    lParams;
    cl_platform_id        platform;
    cl_device_id          device;
    cl_context            context;
    cl_command_queue      commandQueue;
    int                   dataType;
    uint64_t              inputBufferSize;
    uint64_t              outputBufferSize;
};

typedef enum vkfft_transform_dir {
    VKFFT_FORWARD_TRANSFORM    = -1,
    VKFFT_BACKWARD_TRANSFORM   =  1
};

typedef struct interfaceFFTPlan interfaceFFTPlan;

// Interface functions for plan creation
interfaceFFTPlan* vkfftCreateDefaultFFTPlan(cl_context ctx);
interfaceFFTPlan* vkfftCreateR2CFFTPlan(cl_context ctx);

// Interface function for modifying the FFT plan details
void vkfftSetFFTPlanBufferSizes(interfaceFFTPlan* plan);
void vkfftSetFFTPlanDataType(interfaceFFTPlan* plan, int dataType);
void vkfftSetFFTPlanSize(interfaceFFTPlan* plan, size_t lengths[3]);

// Interface functions to make the library compatible with other conventional FFT libraries
VkFFTResult vkfftBakeFFTPlan(interfaceFFTPlan* plan);
VkFFTResult vkfftEnqueueTransform(interfaceFFTPlan* plan, cl_mem* input, cl_mem* dst);
void vkfftDestroyFFTPlan(interfaceFFTPlan* plan);
