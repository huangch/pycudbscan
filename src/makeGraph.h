#ifndef _MAKE_GRAPH_H_
#define _MAKE_GRAPH_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "breadthFirstSearch.h"

using namespace std;

struct Graph {
  long unsigned int *nodes;
  int *edges;
  int totalEdges;
};

void makeGraph(int NUM_BLOCKS, int BLOCK_THREADS, const float *dataPts,
               int numPoints, int minPts, float R, Graph *distGraph,
               int **clusterType);

__global__ void fillNodes(int minPts, float R, int numPoints, int dataDim, float *d_dataPts, 
                          long unsigned int *dNodes, int *dClusterType);

__global__ void fillEdges(int numPoints, int dataDim, float R, float *d_dataPts,
                          long unsigned int *dNodes, int *dEdges);

__device__ __host__ float euclidean_distance(float *p1, float *p2, int dataDim);

#endif
