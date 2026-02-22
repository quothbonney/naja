cmake_minimum_required(VERSION 3.18)
project(naja LANGUAGES C CXX CUDA)

enable_testing()
option(NAJA_ENABLE_ENV_DEP_TESTS "Enable tests that require external model data and/or valid Gurobi license" OFF)
option(NAJA_ENABLE_GPU_TESTS "Enable GPU-based sampling correctness tests (require CUDA device)" OFF)

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
set(KISSFFT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/kissfft")

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

add_executable(naja
    src/main.cu
    src/engine/job_sampling.cu
    src/engine/job_bulk.cpp
    src/engine/bounds_filter.cpp
    src/rounding/jacobi.cpp
    src/rounding/dikin_precondition.cpp
    src/rounding/barrier_rotation.cpp
    src/rounding/barrier_schedule.cpp
    src/rounding/dikin_directions.cpp
    src/rounding/inherit.cpp
    src/util/constraint_utils.cpp
    src/pipeline/extra_constraints.cpp
    src/validate/mcmc.cpp
    src/validate/chord_lengths.cpp
    src/validate/bounds_check.cpp
    src/cli/validate_cli.cpp
    src/cli/sample_cli.cpp
    src/cli/sample/common.cpp
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
    src/rounding/plan.cpp
    src/rounding/runtime_warmup.cpp
    src/rounding/schedule_io.cpp
    src/pipeline/model_contract.cpp
    src/pipeline/feasible_start_files.cpp
    src/pipeline/run_manifest.cpp
    src/pipeline/feasible_start_lp.cpp
    ${GPU_SOURCES}
)

target_include_directories(naja PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GPU_DIR}
    ${EIGEN_DIR}
    ${GUROBI_DIR}/include
    ${KISSFFT_DIR}
)

# kissfft is C — build as a separate static lib to avoid CUDA compiler issues
add_library(kissfft STATIC ${KISSFFT_DIR}/kiss_fft.c ${KISSFFT_DIR}/kiss_fftr.c)
target_include_directories(kissfft PUBLIC ${KISSFFT_DIR})
set_target_properties(kissfft PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(naja PRIVATE kissfft)

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
if(NAJA_ENABLE_ENV_DEP_TESTS)
    add_test(NAME test_eflux_smoke COMMAND test_eflux_smoke)
endif()

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
if(NAJA_ENABLE_ENV_DEP_TESTS)
    add_test(NAME test_condition_cli COMMAND test_condition_cli)
endif()

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
    src/rounding/jacobi.cpp
)
target_include_directories(test_iterative_rounding_jacobi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_test(NAME test_iterative_rounding_jacobi COMMAND test_iterative_rounding_jacobi)

add_executable(test_extra_constraint_eps
    tests/test_extra_constraint_eps.cpp
    src/util/constraint_utils.cpp
)
target_include_directories(test_extra_constraint_eps PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_extra_constraint_eps COMMAND test_extra_constraint_eps)

add_executable(test_extra_constraints_mode
    tests/test_extra_constraints_mode.cpp
    src/pipeline/extra_constraints.cpp
)
target_include_directories(test_extra_constraints_mode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
add_test(NAME test_extra_constraints_mode COMMAND test_extra_constraints_mode)

add_executable(test_pair_schedule_io
    tests/test_pair_schedule_io.cpp
    src/rounding/schedule_io.cpp
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
if(NAJA_ENABLE_ENV_DEP_TESTS)
    add_test(NAME test_feasible_start_lp COMMAND test_feasible_start_lp)
endif()

target_compile_options(naja PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcudafe --diag_suppress=20012
        --use_fast_math
    >
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(naja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(naja PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>
    )
endif()

find_package(CUDAToolkit REQUIRED)

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
if(NAJA_ENABLE_ENV_DEP_TESTS)
    add_test(NAME test_feasible_start_files COMMAND test_feasible_start_files)
endif()

add_test(
    NAME test_sample_cli_dry_run
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sample_cli_dry_run.sh
)

# ---- GPU sampling correctness tests (require CUDA device) ----
if(NAJA_ENABLE_GPU_TESTS)
    set(GPU_TEST_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GPU_DIR}
        ${EIGEN_DIR}
    )
    set(GPU_TEST_LIBS CUDA::cudart CUDA::curand CUDA::cublas)
    set(GPU_TEST_OPTS
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcudafe --diag_suppress=20012>
    )

    # Detect CUDA runtime library path for RPATH
    get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION)
    if(_cudart_loc)
        get_filename_component(CUDA_LIB_DIR "${_cudart_loc}" DIRECTORY)
    else()
        set(CUDA_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}")
    endif()

    set(GPU_TEST_TARGETS
        test_gpu_sampling_feasibility
        test_gpu_sampling_cube_moments
        test_gpu_sampling_simplex_moments
        test_gpu_backmap_correctness
        test_gpu_thin_polytope
    )

    add_executable(test_gpu_sampling_feasibility tests/test_gpu_sampling_feasibility.cu ${GPU_SOURCES})
    add_executable(test_gpu_sampling_cube_moments tests/test_gpu_sampling_cube_moments.cu ${GPU_SOURCES})
    add_executable(test_gpu_sampling_simplex_moments tests/test_gpu_sampling_simplex_moments.cu ${GPU_SOURCES})
    add_executable(test_gpu_backmap_correctness tests/test_gpu_backmap_correctness.cu ${GPU_SOURCES})
    add_executable(test_gpu_thin_polytope tests/test_gpu_thin_polytope.cu ${GPU_SOURCES})

    foreach(T ${GPU_TEST_TARGETS})
        target_include_directories(${T} PRIVATE ${GPU_TEST_INCLUDES})
        target_link_libraries(${T} PRIVATE ${GPU_TEST_LIBS})
        target_compile_options(${T} PRIVATE ${GPU_TEST_OPTS})
        set_target_properties(${T} PROPERTIES BUILD_RPATH "${CUDA_LIB_DIR}")
        add_test(NAME ${T} COMMAND ${T})
    endforeach()
endif()

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "GPU dir: ${GPU_DIR}")
message(STATUS "Eigen dir: ${EIGEN_DIR}")
