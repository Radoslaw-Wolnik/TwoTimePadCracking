#pragma once

#include <gtest/gtest.h>
#include "gpu_accelerator.h"
#include <memory>

class GPUAcceleratorTest : public ::testing::Test {
protected:
    void SetUp() override;
    std::unique_ptr<GPUAccelerator> accelerator;
};