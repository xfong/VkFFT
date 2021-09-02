#include <CL/cl.h>
#include <stdio.h>
#include "header.h"

int main() {
    // OpenCL initialization
    VkGPU vkGPU = {};
    cl_int res = CL_SUCCESS;
    cl_uint numPlatforms;
    res = clGetPlatformIDs(0, 0, &numPlatforms);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * numPlatforms);
    if (!platforms) return VKFFT_ERROR_MALLOC_FAILED;
    res = clGetPlatformIDs(numPlatforms, platforms, 0);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    uint64_t k =0;
    for (uint64_t j = 0; j < numPlatforms; j++) {
        cl_uint numDevices;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
        cl_device_id* deviceList = (cl_device_id*) malloc(sizeof(cl_device_id) * numDevices);
        if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
        if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
        for (uint64_t i = 0; i < numDevices; i++) {
            if (k == vkGPU.device_id) {
                vkGPU.platform = platforms[j];
                vkGPU.device = deviceList[i];
                vkGPU.context = clCreateContext(NULL, 1, &vkGPU.device, NULL, NULL, &res);
                if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
                cl_command_queue commandQueue = clCreateCommandQueue(vkGPU.context, vkGPU.device, 0, &res);
                if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
                vkGPU.commandQueue = commandQueue;
                k++;
            }
            else {
                k++;
            }
        }
        free(deviceList);
    }
    free(platforms);

    printf("Begin test...\n");
    // Generate test data
    uint64_t Nx = 4096;
    float* buffer_input = (float*)malloc(sizeof(float) * 2 * Nx);
    printf("    Generate test data...\n");
    if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
    for (uint64_t i = 0; i < 2 * Nx; i++) {
        buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
    }

    printf("    Setting up plan...\n");
    // Setting up FFT calls
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};

    configuration.FFTdim = 1; //FFT dimenstion, 1D, 2D or 3D
    configuration.size[0] = 4096; //FFT size
    uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0]; // Multiply by 2 for complex numbers

    //Device management + code submission
    configuration.platform = &vkGPU.platform;
    configuration.context = &vkGPU.context;

    printf("    Copying data to GPU...\n");
    // Allocate memory on GPU side
    cl_mem buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize, NULL, &res);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_BUFFER;

    // Copy input data to GPU prior to FFT execution
    res = clEnqueueWriteBuffer(vkGPU.commandQueue, buffer, true, 0, sizeof(float)* 2 * Nx, buffer_input, 0, NULL, NULL);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;

    // Prepare the FFT to run on the GPU side
    printf("    Initializing plan...\n");
    VkFFTResult resFFT = initializeVkFFT(&app, configuration); // In-place FFT

    // Prepare to launch FFT
    printf("    Preparing to launch FFT in GPU...\n");
    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = &buffer;
    launchParams.commandQueue = &vkGPU.commandQueue;

    // Execute FFT on the input buffer
    printf("    Launch FFT in GPU...\n");
    resFFT = VkFFTAppend(&app, -1, &launchParams); // Run FFT

    // Wait for GPU to finish computation
    printf("    Waitting for GPU to return...\n");
    res = clFinish(vkGPU.commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;

    // Allocate memory to transfer GPU results back to host
    printf("    Preparing to retrieve FFT result from GPU...\n");
    float* buffer_output = (float*)malloc(sizeof(float) * 2 * Nx);
    if (!buffer_output) return VKFFT_ERROR_MALLOC_FAILED;

    // Transfer GPU results back to host
    printf("    Retrieve FFT result from GPU...\n");
    res = clEnqueueReadBuffer(vkGPU.commandQueue, buffer, true, 0, sizeof(float)* 2 * Nx, buffer_output, 0, NULL, NULL);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_COPY;

    // Free up GPU memory
    printf("    Tear down...\n");
    res = clReleaseMemObject(buffer);
    if (res != CL_SUCCESS) return (int)(res);

    // Free up resources occupied by VkFFT
    deleteVkFFT(&app);

    printf("Done!\n");
    // End program
    return 0;
}
