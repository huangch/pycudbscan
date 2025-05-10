cmake_minimum_required(VERSION 3.18)
project(pycudbscan LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture
# You may need to adjust this for your specific GPU
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Find packages
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Set up the Python module
pybind11_add_module(pycudbscan_core
  src/pybind_wrapper.cpp
  src/main.cu
)

# Link libraries
target_link_libraries(pycudbscan_core PRIVATE ${CUDA_LIBRARIES})

# Set output directory
set_target_properties(pycudbscan_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pycudbscan
)

# Install target
install(TARGETS pycudbscan_core
  LIBRARY DESTINATION pycudbscan
)