# CUDBSCAN

A cross-platform library for GPU-accelerated DBSCAN clustering using CUDA, with both Python and Java interfaces.

## Overview

CUDBSCAN provides interfaces to a CUDA-based implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. The core CUDA code is accessible from both Python (via pybind11) and Java (via JNI).

## Requirements

- CUDA Toolkit
- CMake (3.18+)
- Python 3.x (for Python bindings)
- pybind11 (for Python bindings)
- JDK 11+ (for Java bindings)
- A C++ compiler compatible with your CUDA version

## Project Structure

```
cudbscan/
├── CMakeLists.txt             # Main CMake configuration
├── examples/                  # Example applications
│   ├── python/                # Python examples
│   │   └── example.py         # Python example script
│   └── java/                  # Java examples
│       └── ExampleJava.java   # Java example
├── python/                    # Python implementation
│   ├── setup.py               # Python package setup
│   ├── environment.yml        # Conda environment definition
│   ├── README.md              # Python-specific documentation
│   └── pycudbscan/            # Python package
│       └── __init__.py        # Python interface
├── java/                      # Java implementation
│   ├── build.gradle           # Gradle build configuration
│   ├── README.md              # Java-specific documentation
│   ├── lib/                   # Native library output directory
│   └── com/cudbscan/          # Java package
│       └── CUDBSCANJava.java  # Java wrapper class
└── src/                       # Core CUDA and binding code
    ├── cuda/                  # CUDA implementation
    │   └── main.cu            # CUDA kernels (dummy implementation)
    ├── python/                # Python binding code
    │   └── pybind_wrapper.cpp # pybind11 wrapper for Python
    └── java/                  # Java binding code
        └── java_wrapper.cpp   # JNI wrapper for Java
```

## Building and Installation

### Using CMake (Recommended)

CMake is used to build both Python and Java bindings in one go:

```bash
# Clone the repository
git clone https://github.com/yourusername/cudbscan.git
cd cudbscan

# Create and activate a build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build everything (Python & Java)
cmake --build . --target build_all

# Run examples (if desired)
cmake --build . --target run_python_example
cmake --build . --target run_java_example
```

### CMake Build Options

You can customize the build with these CMake options:

```bash
# Build only Python bindings
cmake -DBUILD_JAVA_BINDINGS=OFF ..

# Build only Java bindings
cmake -DBUILD_PYTHON_BINDINGS=OFF ..

# Don't build examples
cmake -DBUILD_EXAMPLES=OFF ..
```

### Using Conda (Python only)

```bash
# Create a new conda environment
conda create -n pycudbscan python=3.9
conda activate pycudbscan

# Install required packages
conda install -c conda-forge pybind11 numpy cmake
conda install -c nvidia cuda-toolkit

# Clone the repository
git clone https://github.com/yourusername/cudbscan.git
cd cudbscan

# Build and install the Python package
cd python
pip install -e .
```

### Using Gradle (Java only)

```bash
# Build the Java library
cd java
gradle build

# Run the example
gradle run
```

## Usage Examples

### Python

```python
import numpy as np
from pycudbscan import cuda_dummy_function

# Create sample data
data = np.random.rand(1000).astype(np.float32)

# Call the dummy CUDA function
result = cuda_dummy_function(data)
print(result)
```

### Java

```java
import com.cudbscan.CUDBSCANJava;

public class Example {
    public static void main(String[] args) {
        // Create an instance of the CUDBSCANJava class
        CUDBSCANJava cudbscan = new CUDBSCANJava();
        
        // Check if CUDA is available
        boolean cudaAvailable = cudbscan.checkCudaAvailable();
        System.out.println("CUDA available: " + cudaAvailable);
        
        if (cudaAvailable) {
            // Create a sample array
            float[] input = new float[5];
            for (int i = 0; i < input.length; i++) {
                input[i] = i * 1.0f;
            }
            
            // Call the dummy function
            float[] output = cudbscan.cudaDummyFunction(input);
            
            // Print the results
            for (int i = 0; i < input.length; i++) {
                System.out.printf("Input: %.1f, Output: %.1f\n", input[i], output[i]);
            }
        }
    }
}
```

## Language-Specific Documentation

- For more details on the Python bindings, see [python/README.md](python/README.md)
- For more details on the Java bindings, see [java/README.md](java/README.md)

## Next Steps

Replace the dummy implementation in `src/cuda/main.cu` with your actual DBSCAN implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.