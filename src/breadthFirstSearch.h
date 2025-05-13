#ifndef BREADTH_FIRST_SEARCH_H_
#define BREADTH_FIRST_SEARCH_H_

#include <cuda.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define Not_Visited -1
#define Core 1
#define Border 0

#define PRINT_LOG 0

#define gpuErrchk(ans) \
  { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char* file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

void identifyCluster(int NUM_NODES, int NUM_BLOCKS, int BLOCK_THREADS,
                     long unsigned int* Va, int* Ea, int** clusterIDs,
                     int** clusterType, int* numClusters);
void BreadthFirstSearch(int NUM_NODES, int NUM_BLOCKS, int BLOCK_THREADS,
                        int source, long unsigned int* Va, int* Ea,
                        int* clusterIDs, int* clusterType, int thisClusterID);

__global__ void BreadthFirstSearchKernel(int NUM_NODES, long unsigned int* Va,
                                         int* Ea, bool* Fa, bool* Xa, int* Ca,
                                         int* dClusterType, bool* done);

#endif
