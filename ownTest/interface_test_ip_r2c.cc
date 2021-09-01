#include "fft_interface.h"
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
    cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * numPlatforms);
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
        cl_device_id* deviceList = (cl_device_id*) malloc(sizeof(cl_device_id) * numDevices);
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
    lengths[0] = 128;
    lengths[1] = 64;
    lengths[1] = 32;
    printf("Setting plan lengths...\n");
    setFFTSize(plan, lengths);
    printf("Baking plan...\n");
    res = BakeFFTPlan(plan);
    if (res != VKFFT_SUCCESS) {
        printf("Unable to bake plan...abort\n");
        return -1;
    }
    printf("Exiting...\n");
    DestroyFFTPlan(plan);

    return 0;
}
