#include "fft_interface.h"
#include <CL/cl.hpp>
#include <iostream>

int main() {
    std::cout << "Starting..." << std::endl;
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
        std::cout << "Unable to get initial list of platforms" << std::endl;
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    }
    cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * numPlatforms);
    if (!platforms) return VKFFT_ERROR_MALLOC_FAILED;
    res = clGetPlatformIDs(numPlatforms, platforms, 0);
    if (res != CL_SUCCESS) {
        std::cout << "Unable to get list of platforms" << std::endl;
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
            std::cout << "Unable to get list of devices on platform" << std::endl;
            return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
        }
        for (uint64_t i = 0; i < numDevices; i++) {
            if (k == device_id) {
                platform = platforms[j];
                device = deviceList[i];
                context = clCreateContext(NULL, 1, &device, NULL, NULL, &res);
                if (res != CL_SUCCESS) {
                    std::cout << "Unable to create context" << std::endl;
                    return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
                }
                commandQueue = clCreateCommandQueue(context, device, 0, &res);
                if (res != CL_SUCCESS) {
                    std::cout << "Unable to create command queue" << std::endl;
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
    auto clDevice = cl::Device(device);
    std::string deviceName;
    res = clDevice.getInfo(CL_DEVICE_NAME, &deviceName);
    if (res != CL_SUCCESS) {
        std::cout << "Unable to get device name" << std::endl;
        return res;
    }
    std::cout << "    Targeting: " << deviceName << std::endl;
    return 0;
}
