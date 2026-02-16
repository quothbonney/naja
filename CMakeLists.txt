cmake_minimum_required(VERSION 3.18)
project(naja LANGUAGES CXX CUDA)

enable_testing()

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
set(GUROBI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/gurobi/linux64")

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
    src/engine/iterative_rounding.cpp
    src/engine/constraint_utils.cpp
    src/engine/chr_diagnostics.cpp
    src/engine/extra_constraints.cpp
    src/cli/sample_cli.cpp
    src/cli/sample/common.cpp
    src/cli/sample/rounding.cpp
    src/cli/sample/cmd_run.cpp
    src/cli/sample/cmd_bulk.cpp
    src/cli/sample/cmd_verify.cpp
    src/cli/sample/cmd_eval_rounding.cpp
    src/cli/sample/cmd_calibrate_rounding.cpp
    src/cli/sample/cmd_inherit_rounding.cpp
    src/cli/sample/cmd_list.cpp
    src/cli/prepare/cmd_prepare.cpp
    src/cli/condition_cli.cpp
    src/cli/condition/eflux.cpp
    src/cli/condition/eflux2.cpp
    src/cli/condition/registry.cpp
    src/conditioning/eflux.cpp
    src/conditioning/eflux2.cpp
    src/conditioning/gpr.cpp
    src/pipeline/model_contract.cpp
    src/pipeline/run_layout.cpp
    src/pipeline/config_io.cpp
    src/pipeline/pair_schedule.cpp
    src/pipeline/feasible_start_files.cpp
    src/pipeline/verify_report.cpp
    src/pipeline/run_manifest.cpp
    src/pipeline/feasible_start_lp.cpp
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
    ${GUROBI_DIR}/include
)

add_executable(test_csv_loader
    tests/test_csv_loader.cpp
)
target_include_directories(test_csv_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_csv_loader COMMAND test_csv_loader)

add_executable(test_eflux_smoke
    tests/test_eflux_smoke.cpp
    src/conditioning/eflux.cpp
)
target_include_directories(test_eflux_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
target_compile_options(test_eflux_smoke PRIVATE -fsanitize=address -fno-omit-frame-pointer -O1 -g)
target_link_options(test_eflux_smoke PRIVATE -fsanitize=address)

add_executable(test_condition_cli
    tests/test_condition_cli.cpp
    src/cli/condition/eflux.cpp
    src/conditioning/eflux.cpp
    src/cli/sample/common.cpp
)
target_include_directories(test_condition_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
target_compile_options(test_condition_cli PRIVATE -O3 -march=native)

add_executable(test_gpr_eflux2
    tests/test_gpr_eflux2.cpp
    src/conditioning/gpr.cpp
)
target_include_directories(test_gpr_eflux2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_test(NAME test_gpr_eflux2 COMMAND test_gpr_eflux2)

add_executable(test_eflux2_bounds
    tests/test_eflux2_bounds.cpp
    src/conditioning/eflux2.cpp
    src/conditioning/gpr.cpp
)
target_include_directories(test_eflux2_bounds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_eflux2_bounds COMMAND test_eflux2_bounds)

add_executable(test_start_feasibility
    tests/test_start_feasibility.cpp
)
target_include_directories(test_start_feasibility PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_start_feasibility COMMAND test_start_feasibility)

add_executable(test_chr_pair_steps
    tests/test_chr_pair_steps.cpp
)
target_include_directories(test_chr_pair_steps PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_test(NAME test_chr_pair_steps COMMAND test_chr_pair_steps)

add_executable(test_iterative_rounding_jacobi
    tests/test_iterative_rounding_jacobi.cpp
    src/engine/iterative_rounding.cpp
)
target_include_directories(test_iterative_rounding_jacobi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_test(NAME test_iterative_rounding_jacobi COMMAND test_iterative_rounding_jacobi)

add_executable(test_extra_constraint_eps
    tests/test_extra_constraint_eps.cpp
    src/engine/constraint_utils.cpp
)
target_include_directories(test_extra_constraint_eps PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_extra_constraint_eps COMMAND test_extra_constraint_eps)

add_executable(test_extra_constraints_mode
    tests/test_extra_constraints_mode.cpp
    src/engine/extra_constraints.cpp
)
target_include_directories(test_extra_constraints_mode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_extra_constraints_mode COMMAND test_extra_constraints_mode)

add_executable(test_pair_schedule_io
    tests/test_pair_schedule_io.cpp
    src/pipeline/pair_schedule.cpp
)
target_include_directories(test_pair_schedule_io PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_pair_schedule_io COMMAND test_pair_schedule_io)

add_executable(test_feasible_start_lp
    tests/test_feasible_start_lp.cpp
    src/pipeline/feasible_start_lp.cpp
)
target_include_directories(test_feasible_start_lp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
    ${GUROBI_DIR}/include
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
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
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
    ${GUROBI_DIR}/lib/libgurobi_c++.a
    ${GUROBI_DIR}/lib/libgurobi130.so
)

target_link_libraries(test_feasible_start_lp PRIVATE
    ${GUROBI_DIR}/lib/libgurobi_c++.a
    ${GUROBI_DIR}/lib/libgurobi130.so
)

set_target_properties(naja PROPERTIES
    BUILD_RPATH "${GUROBI_DIR}/lib"
)
set_target_properties(test_feasible_start_lp PROPERTIES
    BUILD_RPATH "${GUROBI_DIR}/lib"
)

add_executable(test_feasible_start_files
    tests/test_feasible_start_files.cpp
    src/pipeline/feasible_start_files.cpp
    src/pipeline/feasible_start_lp.cpp
    src/pipeline/model_contract.cpp
)
target_include_directories(test_feasible_start_files PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
    ${GUROBI_DIR}/include
)
target_link_libraries(test_feasible_start_files PRIVATE
    ${GUROBI_DIR}/lib/libgurobi_c++.a
    ${GUROBI_DIR}/lib/libgurobi130.so
)
set_target_properties(test_feasible_start_files PROPERTIES
    BUILD_RPATH "${GUROBI_DIR}/lib"
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
