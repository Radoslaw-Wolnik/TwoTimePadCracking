// gpu_accelerator.cpp
#include "gpu_accelerator.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

GPUAccelerator::GPUAccelerator() {
    // Get available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)(platforms[0])(), 
        0 
    };
    context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found");
    }

    // Create a command queue and use the first device
    device = devices[0];
    queue = cl::CommandQueue(context, device);
}

GPUAccelerator::~GPUAccelerator() {}

void GPUAccelerator::initializeKernel() {
    std::string kernel_source = loadKernelSource();

    // Create program from source
    program = cl::Program(context, kernel_source);
    if (program.build({device}) != CL_SUCCESS) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        throw std::runtime_error("Error building OpenCL program: " + build_log);
    }

    // Create kernel
    kernel = cl::Kernel(program, "viterbi_step");
}

std::vector<float> GPUAccelerator::viterbiStepGPU(const std::vector<float>& prevProbs, 
                                                  const std::vector<float>& transProbs, 
                                                  const std::vector<float>& emitProbs) {
    size_t num_states = prevProbs.size();
    size_t num_observations = emitProbs.size() / num_states;

    // Create buffers
    cl::Buffer d_prevProbs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                           sizeof(float) * num_states, (void*)prevProbs.data());
    cl::Buffer d_transProbs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            sizeof(float) * num_states * num_states, (void*)transProbs.data());
    cl::Buffer d_emitProbs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                           sizeof(float) * num_states * num_observations, (void*)emitProbs.data());
    cl::Buffer d_newProbs(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_states * num_observations);

    // Set kernel arguments
    kernel.setArg(0, d_prevProbs);
    kernel.setArg(1, d_transProbs);
    kernel.setArg(2, d_emitProbs);
    kernel.setArg(3, d_newProbs);
    kernel.setArg(4, static_cast<cl_uint>(num_states));
    kernel.setArg(5, static_cast<cl_uint>(num_observations));

    // Execute the kernel
    cl::NDRange global(num_states, num_observations);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);

    // Read the result
    std::vector<float> newProbs(num_states * num_observations);
    queue.enqueueReadBuffer(d_newProbs, CL_TRUE, 0, sizeof(float) * num_states * num_observations, newProbs.data());

    return newProbs;
}

std::string GPUAccelerator::loadKernelSource() {
    std::string kernel_path = "../kernels/viterbi_kernel.cl";
    std::ifstream kernel_file(kernel_path);
    if (!kernel_file.is_open()) {
        throw std::runtime_error("Failed to open kernel file");
    }
    return std::string(std::istreambuf_iterator<char>(kernel_file),
                       std::istreambuf_iterator<char>());
}