#include <cuda_runtime.h>
#include <stdio.h>
// Structure to hold optimal CUDA launch parameters
typedef struct {
   int threadsPerBlock;     // Optimal threads per block
   int maxBlocksPerGrid;    // Maximum blocks per grid
   int warpSize;            // Warp size (typically 32)
   int maxThreadsPerSM;     // Maximum threads per SM
   int sharedMemPerBlock;   // Shared memory per block
   int smCount;             // Number of streaming multiprocessors
   int computeMajor;        // Compute capability major version
   int computeMinor;        // Compute capability minor version
} CudaLaunchParams;
// Global variable to store launch parameters
CudaLaunchParams g_launchParams;
// Flag to track initialization status
bool g_cudaInitialized = false;
extern "C" {
   // Initialize CUDA device and resources with dynamic thread configuration
   bool cuda_init() {
       if (g_cudaInitialized) {
           return true; // Already initialized
       }
       // Check if CUDA device is available
       int deviceCount = 0;
       cudaError_t error = cudaGetDeviceCount(&deviceCount);
       // Handle case where no CUDA devices are found
       if (error != cudaSuccess || deviceCount == 0) {
           fprintf(stderr, "CUDA Error: No CUDA devices found or CUDA not available\n");
           return false;
       }
       // Select the first available device (you can modify this to select a specific device)
       error = cudaSetDevice(0);
       if (error != cudaSuccess) {
           fprintf(stderr, "CUDA Error: Failed to select device: %s\n", cudaGetErrorString(error));
           return false;
       }
       // Get device properties
       cudaDeviceProp deviceProp;
       error = cudaGetDeviceProperties(&deviceProp, 0);
       if (error != cudaSuccess) {
           fprintf(stderr, "CUDA Error: Failed to get device properties: %s\n", cudaGetErrorString(error));
           return false;
       }
       // Store device capabilities
       g_launchParams.computeMajor = deviceProp.major;
       g_launchParams.computeMinor = deviceProp.minor;
       g_launchParams.warpSize = deviceProp.warpSize;
       g_launchParams.maxBlocksPerGrid = deviceProp.maxGridSize[0];
       g_launchParams.maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
       g_launchParams.sharedMemPerBlock = deviceProp.sharedMemPerBlock;
       g_launchParams.smCount = deviceProp.multiProcessorCount;
       // ---------------------------------------------------------------------
       // Dynamic thread block size determination based on architecture
       // ---------------------------------------------------------------------
       int threadsPerBlock;
       // Determine optimal thread count based on compute capability
       if (deviceProp.major >= 7) {
           // Volta, Turing, Ampere, Ada Lovelace, Hopper (7.0+)
           threadsPerBlock = 256;
       }
       else if (deviceProp.major >= 5) {
           // Maxwell and Pascal (5.x, 6.x)
           threadsPerBlock = 192;
       }
       else if (deviceProp.major >= 3) {
           // Kepler (3.x)
           threadsPerBlock = 128;
       }
       else {
           // Older architectures (Fermi and earlier)
           threadsPerBlock = 64;
       }
       // Check against device limits
       threadsPerBlock = (threadsPerBlock > deviceProp.maxThreadsPerBlock)
                         ? deviceProp.maxThreadsPerBlock
                         : threadsPerBlock;
       // Make sure it's a multiple of warp size
       threadsPerBlock = (threadsPerBlock / deviceProp.warpSize) * deviceProp.warpSize;
       // Store the result
       g_launchParams.threadsPerBlock = threadsPerBlock;
       // Print diagnostic info
       printf("CUDA Device Initialized: %s\n", deviceProp.name);
       printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("  SMs: %d, Max threads per SM: %d\n",
              deviceProp.multiProcessorCount, deviceProp.maxThreadsPerMultiProcessor);
       printf("  Optimal threads per block: %d\n", g_launchParams.threadsPerBlock);
       printf("  Warp size: %d\n", g_launchParams.warpSize);
       printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
       printf("  Shared memory per block: %d bytes\n", g_launchParams.sharedMemPerBlock);
       printf("  Total global memory: %.2f GB\n",
              deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
       // Reset any previous error
       cudaGetLastError();
       g_cudaInitialized = true;
       return true;
   }
   // Clean up CUDA resources
   void cuda_cleanup() {
       if (!g_cudaInitialized) {
           return; // Nothing to clean up
       }
       // Reset device to clean up all resources
       cudaDeviceReset();
       // Check for errors
       cudaError_t error = cudaGetLastError();
       if (error != cudaSuccess) {
           fprintf(stderr, "CUDA Error during cleanup: %s\n", cudaGetErrorString(error));
       }
       g_cudaInitialized = false;
   }
   // Helper function to get optimal thread/block configuration
   void getOptimalThreadsAndBlocks(int n_elements, dim3 &gridDim, dim3 &blockDim) {
       if (!g_cudaInitialized) {
           // Fallback values if not initialized
           blockDim.x = 128;
           gridDim.x = (n_elements + blockDim.x - 1) / blockDim.x;
           return;
       }
       // Set thread block dimensions (1D configuration)
       blockDim.x = g_launchParams.threadsPerBlock;
       blockDim.y = 1;
       blockDim.z = 1;
       // Calculate grid dimensions
       int blocksNeeded = (n_elements + blockDim.x - 1) / blockDim.x;
       // Basic 1D grid
       gridDim.x = blocksNeeded;
       gridDim.y = 1;
       gridDim.z = 1;
       // If we exceed maximum grid size in x dimension, use 2D grid
       if (blocksNeeded > g_launchParams.maxBlocksPerGrid) {
           int dimX = g_launchParams.maxBlocksPerGrid;
           int dimY = (blocksNeeded + dimX - 1) / dimX;
           gridDim.x = dimX;
           gridDim.y = dimY;
       }
   }
   // Function signature for the DBSCAN implementation
   // You'll replace this with your own implementation
   bool cuda_dbscan(
       const float* input_data,
       int n_samples,
       int n_features,
       float eps,
       int min_samples,
       int* labels,
       int* core_samples,
       int& n_clusters
   ) {
       // Make sure CUDA is initialized
       if (!g_cudaInitialized && !cuda_init()) {
           return false;
       }
       // Your DBSCAN implementation goes here
       // This is just a placeholder that sets all points to cluster 1
       // Replace with your actual implementation
       n_clusters = 1;
       for (int i = 0; i < n_samples; i++) {
           labels[i] = 1;
           core_samples[i] = 1;
       }
       return true;
   }
}

// #include <cuda_runtime.h>
// #include <stdio.h>

// int dclustplus(double *importedDataset, 
//                int *clusterList, int *coreSamples, int *runningCluster, int *clusterCount, int *noiseCount,
//                int dataset_count);
               
// // A very simple dummy kernel
// __global__ void dummy_kernel(float* input, float* output, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         output[idx] = input[idx] * 2.0f;  // Just multiply by 2
//     }
// }

// // C-style function that will be exposed to Python
// extern "C" {
//     // Simple function to check if CUDA is available
//     bool check_cuda_available() {
//         int device_count;
//         cudaError_t error = cudaGetDeviceCount(&device_count);
//         return (error == cudaSuccess && device_count > 0);
//     }
    
//     // Basic dummy function that runs a simple kernel
//     void cuda_dummy_function(float* input, float* output, int size) {
//         // Allocate device memory
//         float *d_input, *d_output;
//         cudaMalloc(&d_input, size * sizeof(float));
//         cudaMalloc(&d_output, size * sizeof(float));
        
//         // Copy input data to device
//         cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
        
//         // Launch kernel (1D grid of threads)
//         int blockSize = 256;
//         int gridSize = (size + blockSize - 1) / blockSize;
//         dummy_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
        
//         // Copy results back to host
//         cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        
//         // Free device memory
//         cudaFree(d_input);
//         cudaFree(d_output);
//     }
// }