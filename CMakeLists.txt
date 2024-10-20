cmake_minimum_required(VERSION 3.15)
project(TwoTimePadCracker VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(ExternalProject)

# Options
option(BUILD_TESTS "Build the tests" ON)
option(USE_GPU "Enable GPU acceleration" OFF)

# Boost
find_package(Boost 1.71 COMPONENTS graph)
if(NOT Boost_FOUND)
    ExternalProject_Add(
        boost
        URL https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    ExternalProject_Get_Property(boost SOURCE_DIR)
    set(BOOST_ROOT ${SOURCE_DIR})
    set(Boost_INCLUDE_DIRS ${BOOST_ROOT})
    set(Boost_LIBRARY_DIRS ${BOOST_ROOT}/stage/lib)
endif()

# yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    ExternalProject_Add(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    )
    ExternalProject_Get_Property(yaml-cpp INSTALL_DIR)
    set(YAML_CPP_INCLUDE_DIR ${INSTALL_DIR}/include)
    set(YAML_CPP_LIBRARIES ${INSTALL_DIR}/lib/libyaml-cpp.a)
endif()

# cxxopts
ExternalProject_Add(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v2.2.1
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)
ExternalProject_Get_Property(cxxopts INSTALL_DIR)
set(CXXOPTS_INCLUDE_DIR ${INSTALL_DIR}/include)

# spdlog
ExternalProject_Add(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.9.2
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)
ExternalProject_Get_Property(spdlog INSTALL_DIR)
set(SPDLOG_INCLUDE_DIR ${INSTALL_DIR}/include)
set(SPDLOG_LIBRARIES ${INSTALL_DIR}/lib/libspdlog.a)

# OpenCL
find_package(OpenCL REQUIRED)

# Threads
find_package(Threads REQUIRED)

# GTest (for tests)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CXXOPTS_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
)

# Add source files
set(SOURCES
    src/main.cpp
    src/config_manager.cpp
    src/decryptor.cpp
    src/gpu_accelerator.cpp
    src/viterbi_search.cpp
    src/viterbi_visualizer.cpp
    src/language_model.cpp
    src/model_analyzer.cpp
    src/visualizer.cpp
    src/logger.cpp
)

# Create executable
add_executable(two_time_pad_cracker ${SOURCES})

# Link libraries
target_link_libraries(two_time_pad_cracker
    PRIVATE
    ${Boost_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${SPDLOG_LIBRARIES}
    ${OpenCL_LIBRARIES}
    Threads::Threads
)

# Add dependencies
add_dependencies(two_time_pad_cracker boost yaml-cpp cxxopts spdlog)

if(USE_GPU)
    target_compile_definitions(two_time_pad_cracker PRIVATE USE_GPU)
endif()

# Enable warnings
if(MSVC)
    target_compile_options(two_time_pad_cracker PRIVATE /W4 /WX)
else()
    target_compile_options(two_time_pad_cracker PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Copy OpenCL kernel file
configure_file(${CMAKE_SOURCE_DIR}/src/viterbi_kernel.cl
               ${CMAKE_BINARY_DIR}/viterbi_kernel.cl COPYONLY)

# Set include directories for target
target_include_directories(two_time_pad_cracker PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CXXOPTS_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
)

# Set output directory
set_target_properties(two_time_pad_cracker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_executable(unit_tests
        test/main_test.cpp
        test/src/test_config_manager.cpp
        test/src/test_decryptor.cpp
        test/src/test_gpu_accelerator.cpp
        test/src/test_language_model.cpp
        test/src/test_visualizer.cpp
        test/src/test_viterbi_search.cpp
    )
    target_include_directories(unit_tests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/test/include
        ${Boost_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${CXXOPTS_INCLUDE_DIR}
        ${SPDLOG_INCLUDE_DIR}
        ${OpenCL_INCLUDE_DIRS}
    )
    target_link_libraries(unit_tests
        PRIVATE
        GTest::GTest
        GTest::Main
        ${Boost_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${SPDLOG_LIBRARIES}
        ${OpenCL_LIBRARIES}
        Threads::Threads
    )
    add_test(NAME unit_tests COMMAND unit_tests)
endif()

# Installation
install(TARGETS two_time_pad_cracker DESTINATION bin)
install(FILES config/config.yml DESTINATION etc)

# Copy config file to build directory
configure_file(${CMAKE_SOURCE_DIR}/config/config.yml
               ${CMAKE_BINARY_DIR}/config.yml COPYONLY)

# Packaging
set(CPACK_PACKAGE_NAME "TwoTimePadCracker")
set(CPACK_PACKAGE_VENDOR "Your Organization")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Two-Time Pad Cracker")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "TwoTimePadCracker")
include(CPack)