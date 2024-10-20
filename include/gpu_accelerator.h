// gpu_accelerator.h
#pragma once

#include <CL/cl.hpp>
#include <string>
#include <vector>

class GPUAccelerator {
public:
    GPUAccelerator();
    ~GPUAccelerator();

    void initializeKernel();
    std::vector<float> viterbiStepGPU(const std::vector<float>& prevProbs, 
                                      const std::vector<float>& transProbs, 
                                      const std::vector<float>& emitProbs);

private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Device device;

    std::string loadKernelSource();
};