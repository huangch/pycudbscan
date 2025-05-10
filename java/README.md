# Java Bindings for CUDBSCAN

This directory contains the Java bindings for the CUDA DBSCAN implementation.

## Environment Setup

You can set up a development environment using Conda:

```bash
# Create a conda environment with Java dependencies
conda env create -f environment.yml
conda activate jcudbscan
```

This environment includes:
- OpenJDK 11+
- Gradle
- CMake and CUDA toolkit
- Build tools (Ninja, Make)

## Building

The Java bindings can be built in two ways:

### 1. Using CMake

```bash
# From the project root
mkdir build && cd build
cmake ..
cmake --build . --target java_build
```

### 2. Using Gradle

If you prefer Gradle for Java development:

```bash
# From the java directory
cd java
gradle build
```

Note: When using Gradle, you still need to build the native library with CMake first:

```bash
mkdir -p build && cd build
cmake -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_JAVA_BINDINGS=ON ..
cmake --build . --target cudbscan_java
cd ..
```

## Directory Structure

```
java/
├── build.gradle        # Gradle build configuration
├── lib/                # Native library output directory
├── README.md           # This file
└── com/cudbscan/       # Java package
    └── CUDBSCANJava.java  # Java wrapper class
```

## Usage

After building, you can use the Java bindings as follows:

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

## API Reference

### `boolean checkCudaAvailable()`

Checks if CUDA is available on the current system.

**Returns:**
- `boolean`: `true` if CUDA is available, `false` otherwise.

### `float[] cudaDummyFunction(float[] input)`

A dummy CUDA function that multiplies each element of the input array by 2.

**Parameters:**
- `input`: 1D array of float values.

**Returns:**
- A new 1D array with the same length as input, with each element multiplied by 2.

## Running with Gradle

To run the example with Gradle:

```bash
cd java
gradle run
```

This will automatically set the `java.library.path` to include the `lib` directory containing the native library.

## Integration with Maven

If you prefer using Maven instead of Gradle, you can create a `pom.xml` file based on the Gradle configuration. The important part is to ensure that the native library is on the `java.library.path`.