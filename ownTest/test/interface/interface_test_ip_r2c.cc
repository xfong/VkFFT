#include "vkfft_c_interface.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("Starting...\n");
    // Main OpenCL variables
    cl_platform_id    platform;
    cl_device_id      device;
    cl_context        context;
    cl_command_queue  commandQueue;
    uint64_t          device_id = 0; // ID of GPU to target
    time_t            tm;
    srand((unsigned)time(&tm));
//    srand((unsigned) 62458790);

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
    interfaceFFTPlan* plan = vkfftCreateR2CFFTPlan(context);
    size_t lengths[3];
    lengths[0] = 8;
    lengths[1] = 4;
    lengths[2] = 2;
    printf("Setting plan lengths...\n");
    vkfftSetFFTPlanSize(plan, lengths);
    printf("Baking plan...\n");
    res = vkfftBakeFFTPlan(plan);
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
    cl_mem ifBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBufferSize, NULL, &res);
    if (res != CL_SUCCESS) {
        printf("Failed to allocate buffer for input...aborting\n");
        return -1;
    }
    cl_mem ofBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, outputBufferSize, NULL, &res);
    if (res != CL_SUCCESS) {
        printf("Failed to allocate buffer for output...aborting\n");
        return -1;
    }
    cl_mem ibBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, outputBufferSize, NULL, &res);
    if (res != CL_SUCCESS) {
        printf("Failed to allocate buffer for input (iFFT)...aborting\n");
        return -1;
    }
    cl_mem obBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBufferSize, NULL, &res);
    if (res != CL_SUCCESS) {
        printf("Failed to allocate buffer for output (iFFT)...aborting\n");
        return -1;
    }

    // First data set
    printf("Generating first data set on host side...\n");
    float tmpFloat = (float)(rand());
    input1[0] = (float)(rand()) / (float)(RAND_MAX);
//    input1[0] = (float)(2534.089);
    for (uint64_t idx = 1; idx < inputElements; idx++) {
        input1[idx] = (float)(0.0);
    }

#if(__DEBUG__>5)
    for (uint64_t idx = 0; idx < inputElements; idx++) {
        printf("  Input[%d] = %e \n", idx, input1[idx]);
    }
#endif

    printf("Generating first data set (output) on host side...\n");
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        output1[idx] = (float)(0.0);
    }

#if(__DEBUG__>5)
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        printf("      Output[%d] = %e \n", idx, output1[idx]);
    }
#endif

    // Copy CPU data over to GPU
    printf("Transferring first data set to GPU side...\n");
    res = clEnqueueWriteBuffer(commandQueue, ifBuf, true, 0, inputBufferSize, input1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        vkfftDestroyFFTPlan(plan);
        printf("Failed to write into buffer for input...aborting\n");
        return -1;
    }
    res = clEnqueueWriteBuffer(commandQueue, ofBuf, true, 0, outputBufferSize, output1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to write into buffer for output...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }

    // Execute FFT
    res = vkfftEnqueueTransform(plan, VKFFT_FORWARD_TRANSFORM, &ifBuf, &ofBuf);
    if (res != CL_SUCCESS) {
        vkfftDestroyFFTPlan(plan);
        printf("Failed to execute forward FFT...\n");
    }

    // Read results back from buffer and print to screen
    res = clFinish(commandQueue);
    if (res != CL_SUCCESS) {
        printf("Failed to wait for queue to clear...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    res = clEnqueueReadBuffer(commandQueue, ofBuf, true, 0, outputBufferSize, output1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to read out buffer for output...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    printf("\n  Results of forward FFT\n");
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        printf("    FFT result [%d]: %15.13e \n", idx, output1[idx]);
    }

    // Execute iFFT
    res = vkfftEnqueueTransform(plan, VKFFT_BACKWARD_TRANSFORM, &ofBuf, &obBuf);
    if (res != CL_SUCCESS) {
        vkfftDestroyFFTPlan(plan);
        printf("Failed to execute backward FFT...\n");
    }

    // Read results back from buffer and print to screen
    res = clFinish(commandQueue);
    if (res != CL_SUCCESS) {
        printf("Failed to wait for queue to clear...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    res = clEnqueueReadBuffer(commandQueue, obBuf, true, 0, inputBufferSize, input1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to read out buffer for output (iFFT)...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    printf("\n    Result of backward FFT\n");
    for (uint64_t idx = 0; idx < inputElements; idx++) {
        printf("    iFFT result [%d]: %15.13e \n", idx, input1[idx]);
    }

    // Second data set
    printf("Generating second data set on host side...\n");
    for (uint64_t idx = 1; idx < inputElements; idx++) {
        input1[idx] = input1[0];
    }

    printf("Generating second data set (output) on host side...\n");
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        output1[idx] = (float) rand() / (float)(RAND_MAX);
    }

    // Copy CPU data over to GPU
    printf("Transferring first data set to GPU side...\n");
    res = clEnqueueWriteBuffer(commandQueue, ifBuf, true, 0, inputBufferSize, input1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        vkfftDestroyFFTPlan(plan);
        printf("Failed to write into buffer for input...aborting\n");
        return -1;
    }
    res = clEnqueueWriteBuffer(commandQueue, ofBuf, true, 0, outputBufferSize, output1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to write into buffer for output...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }

    // Execute FFT
    res = vkfftEnqueueTransform(plan, VKFFT_FORWARD_TRANSFORM, &ifBuf, &ofBuf);
    if (res != CL_SUCCESS) {
        vkfftDestroyFFTPlan(plan);
        printf("Failed to execute forward FFT...\n");
    }

    // Read results back from buffer and print to screen
    res = clFinish(commandQueue);
    if (res != CL_SUCCESS) {
        printf("Failed to wait for queue to clear...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    res = clEnqueueReadBuffer(commandQueue, ofBuf, true, 0, outputBufferSize, output1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to read out buffer for output...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    printf("\n  Results of forward FFT\n");
    for (uint64_t idx = 0; idx < outputElements; idx++) {
        printf("    FFT result [%d]: %15.13e \n", idx, output1[idx]);
    }

    // Execute iFFT
    res = vkfftEnqueueTransform(plan, VKFFT_BACKWARD_TRANSFORM, &ofBuf, &obBuf);
    if (res != CL_SUCCESS) {
        vkfftDestroyFFTPlan(plan);
        printf("Failed to execute backward FFT...\n");
    }

    // Read results back from buffer and print to screen
    res = clFinish(commandQueue);
    if (res != CL_SUCCESS) {
        printf("Failed to wait for queue to clear...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    res = clEnqueueReadBuffer(commandQueue, obBuf, true, 0, inputBufferSize, input1, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        printf("Failed to read out buffer for output (iFFT)...aborting\n");
        vkfftDestroyFFTPlan(plan);
        return -1;
    }
    printf("\n    Result of backward FFT\n");
    for (uint64_t idx = 0; idx < inputElements; idx++) {
        printf("    iFFT result [%d]: %15.13e \n", idx, input1[idx]);
    }

    // Exiting...
    printf("Exiting...\n");
    vkfftDestroyFFTPlan(plan);
    res = clReleaseMemObject(ifBuf);
    if (res != CL_SUCCESS) {
        printf("Failed to release input buffer...aborting\n");
        return -1;
    }
    res = clReleaseMemObject(ofBuf);
    if (res != CL_SUCCESS) {
        printf("Failed to release output buffer...aborting\n");
        return -1;
    }
    res = clReleaseMemObject(ibBuf);
    if (res != CL_SUCCESS) {
        printf("Failed to release input (iFFT) buffer...aborting\n");
        return -1;
    }
    res = clReleaseMemObject(obBuf);
    if (res != CL_SUCCESS) {
        printf("Failed to release output (iFFT) buffer...aborting\n");
        return -1;
    }
    free(input1);
    free(output1);

    return 0;
}
