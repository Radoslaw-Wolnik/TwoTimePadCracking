// test_gpu_acc.cpp
#include <gtest/gtest.h>
#include "gpu_accelerator.h"
#include <vector>
#include <cmath>

class GPUAcceleratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        accelerator = std::make_unique<GPUAccelerator>();
        accelerator->initializeKernel();
    }

    std::unique_ptr<GPUAccelerator> accelerator;
};

TEST_F(GPUAcceleratorTest, TestViterbiStepGPU) {
    std::vector<float> prevProbs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> transProbs = {0.1f, 0.2f, 0.3f, 0.4f,
                                     0.2f, 0.3f, 0.4f, 0.1f,
                                     0.3f, 0.4f, 0.1f, 0.2f,
                                     0.4f, 0.1f, 0.2f, 0.3f};
    std::vector<float> emitProbs = {0.5f, 0.5f, 0.5f, 0.5f,
                                    0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> result = accelerator->viterbiStepGPU(prevProbs, transProbs, emitProbs);

    ASSERT_EQ(result.size(), 8);  // 4 states * 2 observations

    // Check if the results are reasonable (non-negative and sum to approximately 1)
    float sum = 0.0f;
    for (float prob : result) {
        EXPECT_GE(prob, 0.0f);
        sum += prob;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST_F(GPUAcceleratorTest, TestLargeInput) {
    const int num_states = 1000;
    const int num_observations = 100;

    std::vector<float> prevProbs(num_states, 1.0f / num_states);
    std::vector<float> transProbs(num_states * num_states, 1.0f / num_states);
    std::vector<float> emitProbs(num_states * num_observations, 1.0f / num_observations);

    std::vector<float> result = accelerator->viterbiStepGPU(prevProbs, transProbs, emitProbs);

    ASSERT_EQ(result.size(), num_states * num_observations);

    // Check if the results are reasonable
    for (float prob : result) {
        EXPECT_GE(prob, 0.0f);
        EXPECT_LE(prob, 1.0f);
    }
}