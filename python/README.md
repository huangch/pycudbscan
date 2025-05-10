# Python Bindings for CUDBSCAN

This directory contains the Python bindings for the CUDA DBSCAN implementation.

## Building

The Python bindings can be built in several ways:

### 1. Using CMake

```bash
# From the project root
mkdir build && cd build
cmake ..
cmake --build . --target python_install
```

### 2. Using Pip

```bash
# From the python directory
cd python
pip install -e .
```

### 3. Using Conda

```bash
# Create a conda environment with Python dependencies
conda env create -f environment.yml
conda activate pycudbscan

# Now you can use the package directly
```

This environment includes:
- Python 3.9
- pybind11
- NumPy
- CMake and CUDA toolkit
- Development tools (pytest, JupyterLab)

## Directory Structure

```
python/
├── environment.yml     # Conda environment definition
├── pycudbscan/         # Python package
│   └── __init__.py     # Python interface
├── README.md           # This file
└── setup.py            # Python package setup
```

## Usage

After installation, you can use the Python bindings as follows:

```python
import numpy as np
from pycudbscan import cuda_dummy_function, check_cuda_available

# Check if CUDA is available
if check_cuda_available():
    # Create a sample array
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # Call the dummy function
    result = cuda_dummy_function(data)
    
    # Print the results
    for i in range(len(data)):
        print(f"Input: {data[i]}, Output: {result[i]}")
else:
    print("CUDA is not available")
```

## API Reference

### `check_cuda_available()`

Checks if CUDA is available on the current system.

**Returns:**
- `bool`: `True` if CUDA is available, `False` otherwise.

### `cuda_dummy_function(data)`

A dummy CUDA function that multiplies each element of the input array by 2.

**Parameters:**
- `data (numpy.ndarray)`: 1D array of float32 values.

**Returns:**
- `numpy.ndarray`: 1D array with the same shape as input, with each element multiplied by 2.

## Performance Notes

The Python bindings use NumPy arrays for data transfer between Python and CUDA. This approach allows for efficient memory management and minimizes data copying.

## Dependencies

- NumPy: For array handling
- pybind11: For C++ to Python bindings
- CUDA Toolkit: For GPU computation