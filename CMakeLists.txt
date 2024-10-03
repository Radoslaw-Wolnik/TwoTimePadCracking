cmake_minimum_required(VERSION 3.10)
project(TwoTimePadCracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

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

# Include directories
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CXXOPTS_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
)

# Add your source files
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

add_executable(two_time_pad_cracker ${SOURCES})

# Link libraries
target_link_libraries(two_time_pad_cracker
    ${Boost_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${SPDLOG_LIBRARIES}
    ${OpenCL_LIBRARIES}
)

# Add dependencies
add_dependencies(two_time_pad_cracker boost yaml-cpp cxxopts spdlog)

# Copy OpenCL kernel file
configure_file(${CMAKE_SOURCE_DIR}/src/viterbi_kernel.cl
               ${CMAKE_BINARY_DIR}/viterbi_kernel.cl COPYONLY)

# Set include directories for target
target_include_directories(two_time_pad_cracker PRIVATE
    ${CMAKE_SOURCE_DIR}/src
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