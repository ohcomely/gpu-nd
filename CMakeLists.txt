cmake_minimum_required(VERSION 3.18)
project(NestedDissection LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust as needed for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Add debug flags for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -DDEBUG")

# Source files
set(SOURCES
    bin/structs.cpp
    bin/mtx_reader.cpp
    bin/nested_dissection.cu
    bin/main.cpp
)

# Create executable
add_executable(nested_dissection ${SOURCES})

# Link libraries
target_link_libraries(nested_dissection 
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
)

# Set properties for CUDA compilation
set_property(TARGET nested_dissection PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Ensure proper CUDA linking
set_target_properties(nested_dissection PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optional: Add installation rules
install(TARGETS nested_dissection DESTINATION bin)

# Optional: Add test target if you have tests
# enable_testing()
# add_subdirectory(tests)