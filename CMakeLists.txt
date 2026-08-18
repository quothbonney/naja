cmake_minimum_required(VERSION 3.24)
project(naja LANGUAGES C CXX CUDA)

enable_testing()
option(NAJA_ENABLE_ENV_DEP_TESTS "Enable tests that require external model data and/or valid Gurobi license" OFF)
option(NAJA_ENABLE_GPU_TESTS "Enable GPU-based sampling correctness tests (require CUDA device)" OFF)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# CUDA architectures. Default to `native` (auto-detect the build host's GPU) so
# naja rebuilds correctly per cluster with no edits: A100->sm_80, H100->sm_90,
# B300->sm_100/103. Override for a portable fat binary, e.g.:
#   cmake -DCMAKE_CUDA_ARCHITECTURES="80;90;100;103" ...
# (Note: CUDA 13 dropped Volta sm_70; Blackwell needs CUDA >= 12.8.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Dependency locations. A local extern/ tree remains supported for existing
# deployments, while fresh clones can use standard system installations.
set(GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu")
set(KISSFFT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/kissfft")

set(NAJA_EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen3" CACHE PATH
    "Directory containing the Eigen headers (for example /usr/include/eigen3)")
find_path(EIGEN_DIR Eigen/Core
    HINTS "${NAJA_EIGEN_ROOT}" "$ENV{EIGEN3_ROOT}"
    PATH_SUFFIXES include/eigen3 eigen3)
if(NOT EIGEN_DIR)
    message(FATAL_ERROR
        "Eigen3 headers were not found. Install Eigen3 or configure -DNAJA_EIGEN_ROOT=/path/to/eigen3.")
endif()

set(NAJA_GUROBI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/gurobi/linux64" CACHE PATH
    "Gurobi installation root (contains include/ and lib/)")
find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
    HINTS "${NAJA_GUROBI_ROOT}" "$ENV{GUROBI_HOME}"
    PATH_SUFFIXES include)
find_library(GUROBI_CXX_LIBRARY NAMES gurobi_c++
    HINTS "${NAJA_GUROBI_ROOT}" "$ENV{GUROBI_HOME}"
    PATH_SUFFIXES lib)
find_library(GUROBI_LIBRARY NAMES gurobi130
    HINTS "${NAJA_GUROBI_ROOT}" "$ENV{GUROBI_HOME}"
    PATH_SUFFIXES lib)
if(NOT GUROBI_INCLUDE_DIR OR NOT GUROBI_CXX_LIBRARY OR NOT GUROBI_LIBRARY)
    message(FATAL_ERROR
        "Gurobi 13 development files were not found. Install Gurobi 13 and set GUROBI_HOME, or configure -DNAJA_GUROBI_ROOT=/path/to/gurobi.")
endif()
get_filename_component(GUROBI_LIBRARY_DIR "${GUROBI_LIBRARY}" DIRECTORY)

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
    src/npz.cpp
    src/pipeline/model_io.cpp
    src/engine/job_sampling.cu
    src/engine/job_bulk.cpp
    src/engine/bounds_filter.cpp
    src/rounding/jacobi.cpp
    src/rounding/dikin_precondition.cpp
    src/rounding/barrier_rotation.cu
    src/rounding/barrier_schedule.cpp
    src/rounding/dikin_directions.cu
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
    ${GUROBI_INCLUDE_DIR}
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

add_executable(test_mcmc_interleaved_layout
    tests/test_mcmc_interleaved_layout.cpp
    src/validate/mcmc.cpp
)
target_include_directories(test_mcmc_interleaved_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
    ${KISSFFT_DIR}
)
target_link_libraries(test_mcmc_interleaved_layout PRIVATE kissfft)
add_test(NAME test_mcmc_interleaved_layout COMMAND test_mcmc_interleaved_layout)

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
    ${GUROBI_INCLUDE_DIR}
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
find_package(ZLIB REQUIRED)  # for .npz (zip) member decompression in src/npz.cpp

target_link_libraries(naja PRIVATE
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    ZLIB::ZLIB
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
)

target_link_libraries(test_feasible_start_lp PRIVATE
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
)

set_target_properties(naja PROPERTIES
    BUILD_RPATH "${GUROBI_LIBRARY_DIR}"
)
set_target_properties(test_feasible_start_lp PROPERTIES
    BUILD_RPATH "${GUROBI_LIBRARY_DIR}"
)

add_executable(test_feasible_start_files
    tests/test_feasible_start_files.cpp
    src/pipeline/feasible_start_files.cpp
    src/pipeline/feasible_start_lp.cpp
    src/pipeline/model_contract.cpp
    src/pipeline/model_io.cpp
    src/npz.cpp
)
target_include_directories(test_feasible_start_files PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
    ${GUROBI_INCLUDE_DIR}
)
target_link_libraries(test_feasible_start_files PRIVATE
    ZLIB::ZLIB
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
)
set_target_properties(test_feasible_start_files PROPERTIES
    BUILD_RPATH "${GUROBI_LIBRARY_DIR}"
)
if(NAJA_ENABLE_ENV_DEP_TESTS)
    add_test(NAME test_feasible_start_files COMMAND test_feasible_start_files)
endif()

add_test(
    NAME test_sample_cli_dry_run
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sample_cli_dry_run.sh
)

add_executable(test_npz_reader
    tests/test_npz_reader.cpp
    src/npz.cpp
)
target_include_directories(test_npz_reader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_DIR}
)
target_link_libraries(test_npz_reader PRIVATE ZLIB::ZLIB)
add_test(
    NAME test_npz_reader
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_npz_reader.sh $<TARGET_FILE:test_npz_reader>
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
