cmake_minimum_required(VERSION 3.18)
project(naja LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# CUDA architectures (A100)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Paths to local source copies
set(GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu")
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen3")

if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "Eigen3 not found at ${EIGEN_DIR}. Please run scripts/setup_extern.sh")
endif()

# GPU source files
set(GPU_SOURCES
    ${GPU_DIR}/chr.cu
    ${GPU_DIR}/dmatrix.cu
    ${GPU_DIR}/dvector.cu
    ${GPU_DIR}/device_utils.cu
    ${GPU_DIR}/backmap.cu
)

# Demo executable
add_executable(naja
    src/main.cu
    src/engine/job_sampling.cu
    src/engine/job_bulk.cpp
    src/engine/bounds_filter.cpp
    src/cli/sample_cli.cpp
    src/pipeline/model_contract.cpp
    src/pipeline/run_layout.cpp
    src/pipeline/config_io.cpp
    src/pipeline/verify_report.cpp
    src/pipeline/run_manifest.cpp
    ${GPU_SOURCES}
)

# Coloring Tool (temporarily disabled)
#add_executable(cu_coloring
#    src/coloring/main.cu
#    src/coloring/soft_kernels.cu
#    src/gpu/device_utils.cu
#    src/gpu/dmatrix.cu
#    src/gpu/dvector.cu
#)

# Include directories
target_include_directories(naja PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GPU_DIR}
    ${EIGEN_DIR}
)

# Coloring tool includes (disabled)
#target_include_directories(cu_coloring PRIVATE
#    ${CMAKE_CURRENT_SOURCE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/src
#    ${GPU_DIR}
#    ${EIGEN_DIR}
#)

# CUDA compile options
target_compile_options(naja PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcudafe --diag_suppress=20012
        --use_fast_math
    >
)

# Coloring tool compile options (disabled)
#target_compile_options(cu_coloring PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:
#        --expt-relaxed-constexpr
#        -Xcudafe --diag_suppress=20012
#        --use_fast_math
#    >
#)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(naja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    #target_compile_options(cu_coloring PRIVATE
    #    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    #    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    #)
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(naja PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>
    )
    #target_compile_options(cu_coloring PRIVATE
    #    $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>
    #)
endif()

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Link libraries
target_link_libraries(naja PRIVATE
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
)

# Coloring tool links (disabled)
#target_link_libraries(cu_coloring PRIVATE
#    CUDA::cudart
#    CUDA::curand
#    CUDA::cublas
#)

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "GPU dir: ${GPU_DIR}")
message(STATUS "Eigen dir: ${EIGEN_DIR}")
