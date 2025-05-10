# PyCUDBSCAN

A Python library for GPU-accelerated DBSCAN clustering using CUDA.

## Overview

PyCUDBSCAN provides a Python interface to a CUDA-based implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. This project contains the basic structure needed to build and wrap CUDA code with pybind11.

## Requirements

- CUDA Toolkit
- CMake (3.18+)
- Python 3.x
- pybind11
- A C++ compiler compatible with your CUDA version

## Project Structure

```
pycudbscan/
├── CMakeLists.txt        # Main CMake configuration
├── setup.py              # Python package setup
├── src/
│   ├── main.cu           # CUDA kernels (dummy implementation)
│   └── pybind_wrapper.cpp # pybind11 wrapper code
└── pycudbscan/           # Python package
    └── __init__.py       # Python interface
```

## Installation

### Using Conda

```bash
# Create a new conda environment
conda create -n pycudbscan python=3.9
conda activate pycudbscan

# Install required packages
conda install -c conda-forge pybind11 numpy cmake
conda install -c nvidia cuda-toolkit

# Clone the repository
git clone https://github.com/yourusername/pycudbscan.git
cd pycudbscan

# Build and install the package
pip install -e .
```

### Manual Build

```bash
# Clone the repository
git clone https://github.com/yourusername/pycudbscan.git
cd pycudbscan

# Create and activate a build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# Install
pip install -e ..
```

## Usage Example

```python
import numpy as np
from pycudbscan import cuda_dummy_function

# Create sample data
data = np.random.rand(1000).astype(np.float32)

# Call the dummy CUDA function
result = cuda_dummy_function(data)
print(result)
```

## Next Steps

Replace the dummy implementation in `src/main.cu` with your actual DBSCAN implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.