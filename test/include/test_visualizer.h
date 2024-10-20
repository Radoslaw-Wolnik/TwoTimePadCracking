#pragma once

#include <gtest/gtest.h>
#include "visualizer.h"
#include <memory>

class VisualizerTest : public ::testing::Test {
protected:
    void SetUp() override;
    std::unique_ptr<Visualizer> visualizer;
};