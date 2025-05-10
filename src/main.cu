#include <cuda_runtime.h>
#include <stdio.h>

// A very simple dummy kernel
__global__ void dummy_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;  // Just multiply by 2
    }
}

// C-style function that will be exposed to Python
extern "C" {
    // Simple function to check if CUDA is available
    bool check_cuda_available() {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess && device_count > 0);
    }
    
    // Basic dummy function that runs a simple kernel
    void cuda_dummy_function(float* input, float* output, int size) {
        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));
        
        // Copy input data to device
        cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch kernel (1D grid of threads)
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        dummy_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
        
        // Copy results back to host
        cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }
}