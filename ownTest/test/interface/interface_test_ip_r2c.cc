#include "vkfft_c_interface.h"
#include <CL/cl.h>
#include <stdio.h>

int main() {
    printf("Starting...\n");
    // Main OpenCL variables
    cl_platform_id    platform;
    cl_device_id      device;
    cl_context        context;
    cl_command_queue  commandQueue;
    uint64_t          device_id = 0; // ID of GPU to target

    // Setting up GPU OpenCL device
    cl_int res = CL_SUCCESS;
    cl_uint numPlatforms;
    cl_platform_id tmp;
    res = clGetPlatformIDs(1, &tmp, &numPlatforms);
    if (res != CL_SUCCESS) {
        printf("Unable to get initial list of platforms\n");
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    }
    cl_platform_id* platforms = (cl_platform_id*) calloc(1, sizeof(cl_platform_id) * numPlatforms);
    if (!platforms) return VKFFT_ERROR_MALLOC_FAILED;
    res = clGetPlatformIDs(numPlatforms, platforms, 0);
    if (res != CL_SUCCESS) {
        printf("Unable to get list of platforms\n");
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    }
    uint64_t k =0;
    for (uint64_t j = 0; j < numPlatforms; j++) {
        cl_uint numDevices;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
        cl_device_id* deviceList = (cl_device_id*) calloc(1, sizeof(cl_device_id) * numDevices);
        if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
        if (res != CL_SUCCESS) {
            printf("Unable to get list of devices on platform\n");
            return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
        }
        for (uint64_t i = 0; i < numDevices; i++) {
            if (k == device_id) {
                platform = platforms[j];
                device = deviceList[i];
                context = clCreateContext(NULL, 1, &device, NULL, NULL, &res);
                if (res != CL_SUCCESS) {
                    printf("Unable to create context\n");
                    return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
                }
                commandQueue = clCreateCommandQueue(context, device, 0, &res);
                if (res != CL_SUCCESS) {
                    printf("Unable to create command queue\n");
                    return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
                }
                k++;
            }
            else {
                k++;
            }
        }
        free(deviceList);
    }
    free(platforms);
    char deviceName[512];
    res = clGetDeviceInfo(device, CL_DEVICE_NAME, 512, &deviceName[0], NULL);
    if (res != CL_SUCCESS) {
        printf("Unable to get device name\n");
        return res;
    }
    printf("    Targeting: %s \n", deviceName);

    // Create the R2C plan...
    printf("Creating plan...\n");
    interfaceFFTPlan* plan = createR2CFFTPlan(context);
    size_t lengths[3];
    lengths[0] = 32;
    lengths[1] = 8;
    lengths[2] = 4;
    printf("Setting plan lengths...\n");
    setFFTSize(plan, lengths);
    printf("Baking plan...\n");
    res = BakeFFTPlan(plan);
    if (res != VKFFT_SUCCESS) {
        printf("Unable to bake plan...abort\n");
        return -1;
    }

    // Create date for testing...
    printf("    Begin testing...\n");

    // Allocate memory
    printf("Allocating memory...\n");
    size_t inputElements    = lengths[0]*lengths[1]*lengths[2];
    size_t outputElements   = (lengths[0] /2 + 1)*lengths[1]*lengths[2];
    size_t inputBufferSize  = sizeof(float)*inputElements;
    size_t outputBufferSize = 2*sizeof(float)*outputElements;

    float* input1  = (float*) calloc(1, inputBufferSize);
    float* output1 = (float*) calloc(1, outputBufferSize);

    // Allocate GPU buffers
    printf("Creating buffers...\n");
    cl_mem iBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBufferSize, NULL, &res);
    if (res != CL_SUCCESS) {
        printf("Failed to allocate buffer for input...aborting\n");
        return -1;
    }
    cl_mem oBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, outputBufferSize, NULL, &res);
    if (res != CL_SUCCESS) {
        printf("Failed to allocate buffer for output...aborting\n");
        return -1;
    }

    // First data set
    printf("Generating first data set on host side...\n");
    input1[0] = (float)(rand()) / (float)(RAND_MAX);
    for (uint64_t idx = 0; idx < inputElements; idx++) {
        input1[idx] = (float)(0.0);
    }

    printf("Generating first data set (output) on host side...\n");
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        output1[idx] = (float)(0.0);
    }

    // Copy CPU data over to GPU
    printf("Transferring first data set to GPU side...\n");
    res = clEnqueueWriteBuffer(commandQueue, iBuf, true, 0, inputBufferSize, input1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to write into buffer for input...aborting\n");
        return -1;
    }
    res = clEnqueueWriteBuffer(commandQueue, oBuf, true, 0, outputBufferSize, output1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to write into buffer for output...aborting\n");
        return -1;
    }

    // Second data set
    printf("Generating second data set on host side...\n");
    for (uint64_t idx = 0; idx < inputElements; idx++) {
        input1[idx] = (float) rand() / (float)(RAND_MAX);
    }

    printf("Generating second data set (output) on host side...\n");
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        output1[idx] = (float) rand() / (float)(RAND_MAX);
    }

    // Exiting...
    printf("Exiting...\n");
    DestroyFFTPlan(plan);
    res = clReleaseMemObject(iBuf);
    if (res != CL_SUCCESS) {
        printf("Failed to release input buffer...aborting\n");
        return -1;
    }
    res = clReleaseMemObject(oBuf);
    if (res != CL_SUCCESS) {
        printf("Failed to release output buffer...aborting\n");
        return -1;
    }
    free(input1);
    free(output1);

    return 0;
}
