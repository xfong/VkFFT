// Interface to vkFFT with convenice functions
#include "fft_interface.h"

// Basic function to return a FFT plan.
// This flow is similar to other FFT libraries such as FFTW, cuFFT, clFFT, rocFFT.
interfaceFFTPlan* createFFTPlan(cl_context ctx) {
    interfaceFFTPlan* plan = new interfaceFFTPlan;

    cl_int res;
    // Grab required information from context given...
    plan->context = ctx;

    // Get device ID from context
    size_t numCount = 0;
    res = clGetContextInfo(plan->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &plan->device, &numCount);
    if (res != CL_SUCCESS) {
        delete(plan);
        return NULL;
    }

    res = clGetDeviceInfo(plan->device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plan->platform, &numCount);
    if (res != CL_SUCCESS) {
        delete(plan);
        return NULL;
    }

    // Create a command queue for the plan
    plan->commandQueue = clCreateCommandQueue(plan->context, plan->device, 0, &res);
    if (res != CL_SUCCESS) {
        delete(plan);
        return NULL;
    }

    // Update internal pointers
    plan->config.platform = &plan->platform;
    plan->config.context = &plan->context;
    plan->config.device = &plan->device;
    plan->lParams.commandQueue = &plan->commandQueue;

    // Default to 3D plan but with all dimensions to be 1
    plan->config.FFTdim  = 3;
    plan->config.size[0] = 1;
    plan->config.size[1] = 1;
    plan->config.size[2] = 1;

    // Default to C2C transform and in-place invervse
    plan->config.performR2C = 0;
    plan->config.inverseReturnToInputBuffer = 0;

    // Default to out-of-place transform
    plan->config.isInputFormatted = 1;

    // Default to float
    // half   = -1
    // float  =  0
    // double =  1
    plan->dataType = 0;

    // Initialize flag to ensure plan is baked before execution can happen
    plan->isBaked = false;
    return plan;
}

// A specialized function to return a FFT plan that computes
// R2C (forward) and C2R (backward) transforms
interfaceFFTPlan* createR2CFFTPlan(cl_context ctx) {
    interfaceFFTPlan* plan = createFFTPlan(ctx);
    plan->config.performR2C = 1;
    plan->config.inverseReturnToInputBuffer = 1;
    return plan;
}

// Interface function to set up the data type for the FFT
void setFFTPlanDataType(interfaceFFTPlan* plan, int dataType) {
    // Default to float
    // half   = -1
    // float  =  0
    // double =  1
    plan->dataType = dataType;
    setFFTPlanBufferSizes(plan);
}

// Interface function to set up the FFT sizes
void setFFTSize(interfaceFFTPlan* plan, size_t lengths[3]) {
    plan->config.size[0] = lengths[0];
    plan->config.size[1] = lengths[1];
    plan->config.size[2] = lengths[2];
    // If the plan was previously baked, we need to clean up the plan
    if (plan->isBaked) {
        plan->app = {};
    }
    plan->isBaked = false;

    ////// Order the lengths of the FFT so it can be "fast"

    // Default to 3D first but set all sizes to 1
    plan->config.FFTdim = 3;
    plan->config.size[0] = 1;
    plan->config.size[1] = 1;
    plan->config.size[2] = 1;

    // Find out the desired dimensionality of the FFT
    if (lengths[0] == 1) { plan->config.FFTdim--; }
    if (lengths[1] == 1) { plan->config.FFTdim--; }
    if (lengths[2] == 1) { plan->config.FFTdim--; }

    // Catch when all entries of lengths[] is 1
    if (plan->config.FFTdim == 0) {
        plan->config.FFTdim = 1; // the FFT has all lengths to be 1
    } else if (plan->config.FFTdim == 1) { // Case where FFT is 1D
        // Find the entry of lengths[] that is not 1 and assign to
        // config.size[0] (the other entries default to 1 from before)
        if (lengths[0] != 1) {
            plan->config.size[0] = lengths[0];
        } else if (lengths[1] != 1) {
            plan->config.size[0] = lengths[1];
        } else {
            plan->config.size[0] = lengths[2];
        }
    } else if (plan->config.FFTdim == 2) { // Case where FFT is 2D
        // Find the entry of lengths[] that is 1 and assign to remaining
        // to config.size[0] and config.size[1] (the remaining entry
        // default to 1 from before)
        if (lengths[0] == 1) {
            plan->config.size[0] = lengths[1];
            plan->config.size[1] = lengths[2];
        } else if (lengths[1] == 1) {
            plan->config.size[0] = lengths[0];
            plan->config.size[1] = lengths[2];
        } else {
            plan->config.size[0] = lengths[0];
            plan->config.size[1] = lengths[1];
        }
    } else { // Case where FFT is 3D
        plan->config.size[0] = lengths[0];
        plan->config.size[1] = lengths[1];
        plan->config.size[2] = lengths[2];
    }
    setFFTPlanBufferSizes(plan);
}

// Function to determine the input and output buffer sizes
void setFFTPlanBufferSizes(interfaceFFTPlan* plan) {
    // Input and output buffer sizes if transform is C2C
    plan->inputBufferSize  = plan->config.size[1] * plan->config.size[2];

    if (plan->dataType < 0) {
        plan->inputBufferSize  *= __SIZEOF_HALF__;
    } else if (plan->dataType > 0) {
        plan->inputBufferSize  *= __SIZEOF_DOUBLE__;
    } else {
        plan->inputBufferSize  *= __SIZEOF_FLOAT__;
    }

    plan->outputBufferSize = plan->inputBufferSize;

    // If plan is already defined as R2C, then we can set the input and output buffer sizes
    // as well as the strides
    if (plan->config.performR2C == 1) {
        plan->inputBufferSize  *= plan->config.size[0];
        plan->outputBufferSize *= 2 * (plan->config.size[0] / 2 + 1);
    } else { // Otherwise, plan is C2C
        plan->inputBufferSize  *= 2 * plan->config.size[0];
        plan->outputBufferSize  = plan->inputBufferSize;
    }

    // Update plan
    plan->config.inputBufferSize = &plan->inputBufferSize;
    plan->config.bufferSize      = &plan->outputBufferSize;
}

// Interface to initializeVkFFT()
// Provide this function so that initialization can be checked prior to
// any execution
VkFFTResult BakeFFTPlan(interfaceFFTPlan* plan) {
    VkFFTResult res;
    res = initializeVkFFT(&plan->app, plan->config);
    if (res == VKFFT_SUCCESS) {
        plan->isBaked = true;
    } else {
        plan->isBaked = false;
    }
    return res;
}

// Interface function to perform a forward FFT.
// This function will ensure the plan is initialized prior to execution.
VkFFTResult executeForwardFFT(interfaceFFTPlan* plan, cl_mem* input, cl_mem* dst) {
    // Set up buffers for input and output so that vkFFT can recognize them
    plan->lParams.inputBuffer = input;
    plan->lParams.buffer = dst;

    VkFFTResult res;
    // Initialize the plan if it is not already initialized
    if (!plan->isBaked) {
        res = BakeFFTPlan(plan);
        if (res != VKFFT_SUCCESS) {
            return res;
        }
    }

    // Plan is guaranteed to be initialized so we launch the execution
    return VkFFTAppend(&plan->app, -1, &plan->lParams);
}

// Interface function to perform a backward FFT.
// This function will ensure the plan is initialized prior to execution.
VkFFTResult executeBackwardFFT(interfaceFFTPlan* plan, cl_mem* input, cl_mem* dst) {
    // Set up buffers for input and output so that vkFFT can recognize them
    if (plan->config.inverseReturnToInputBuffer == 1) {
        plan->lParams.inputBuffer = dst;
        plan->lParams.buffer = input;
    } else {
        plan->lParams.inputBuffer = input;
        plan->lParams.buffer = dst;
    }
    VkFFTResult res;

    // Initialize the plan if it is not already initialized
    if (!plan->isBaked) {
        res = BakeFFTPlan(plan);
        if (res != VKFFT_SUCCESS) {
            return res;
        }
    }

    // Plan is guaranteed to be initialized so we launch the execution
    return VkFFTAppend(&plan->app, 1, &plan->lParams);
}

// Interface function to clean up
void DestroyFFTPlan(interfaceFFTPlan* plan) {
    deleteVkFFT(&plan->app);
    delete(plan);
}
