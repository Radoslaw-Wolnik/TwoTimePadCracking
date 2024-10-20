// main_test.cpp
#include <gtest/gtest.h>

// Include headers for new test files
#include "test_config_manager.h"
#include "test_decryptor.h"
#include "test_gpu_accelerator.h"
#include "test_language_model.h"
#include "test_visualizer.h"
#include "test_viterbi_search.h"


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    
    // You can add any global test environment setup here if needed
    // For example:
    // testing::AddGlobalTestEnvironment(new MyTestEnvironment);
    
    return RUN_ALL_TESTS();
}