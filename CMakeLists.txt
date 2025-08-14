cmake_minimum_required(VERSION 3.18)
project(FastNestedDissection LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA compute capability (adjust based on your GPU)
# Common values: 61 (GTX 1060-1080), 75 (RTX 2080), 86 (RTX 3080-3090), 89 (RTX 4090)
set(CUDA_COMPUTE_CAPABILITY "86" CACHE STRING "CUDA compute capability")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(CUDA_SOURCES
    src/nested_dissection_core.cu
    src/nested_dissection_algorithms.cu
    src/main.cu
)

# Create executable
add_executable(fast_nested_dissection ${CUDA_SOURCES})

# Set CUDA properties
set_target_properties(fast_nested_dissection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA compile options
target_compile_options(fast_nested_dissection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --extended-lambda
        -gencode arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}
    >
)

# C++ compile options
target_compile_options(fast_nested_dissection PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

# Link libraries
target_link_libraries(fast_nested_dissection
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    OpenMP::OpenMP_CXX
)

# Find CUDA libraries manually if not found automatically
if(NOT CUDA_cusparse_LIBRARY)
    find_library(CUDA_cusparse_LIBRARY cusparse HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
endif()

if(NOT CUDA_cublas_LIBRARY)
    find_library(CUDA_cublas_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
endif()

if(NOT CUDA_cusolver_LIBRARY)
    find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
endif()

# Print library locations for debugging
message(STATUS "CUDA Libraries:")
message(STATUS "  cuSPARSE: ${CUDA_cusparse_LIBRARY}")
message(STATUS "  cuBLAS: ${CUDA_cublas_LIBRARY}")
message(STATUS "  cuSOLVER: ${CUDA_cusolver_LIBRARY}")

# Compiler-specific options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # Additional NVIDIA-specific flags can go here
    target_compile_options(fast_nested_dissection PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
endif()

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fast_nested_dissection PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Release configuration (default)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_compile_options(fast_nested_dissection PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
    )
endif()

# Optional: Create a test executable with smaller matrix for quick testing
add_executable(fast_nested_dissection_test ${CUDA_SOURCES})
target_compile_definitions(fast_nested_dissection_test PRIVATE SMALL_TEST_MATRIX=1)
set_target_properties(fast_nested_dissection_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(fast_nested_dissection_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --extended-lambda
        -gencode arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}
    >
)
target_link_libraries(fast_nested_dissection_test
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    OpenMP::OpenMP_CXX
)