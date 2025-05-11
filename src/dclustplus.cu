#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <map>
#include <math.h>
#include <set>
#include <vector>

#include "common.h"
#include "indexing.h"
#include "dbscan.h"

int DATASET_COUNT;

using namespace std;

// int ImportDataset(char const *fname, double *dataset);

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Main CPU function
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
// int main(int argc, char **argv) {
int dclustplus(double *importedDataset, //
               int *d_cluster, int *runningCluster, int *clusterCount, int *noiseCount, // Output
               int dataset_count // Parameters
              ) {
  DATASET_COUNT = dataset_count;
  // *.**
  // char inputFname[500];
  // if (argc != 2) {
  //   fprintf(stderr, "Please provide the dataset file path in the arguments\n");
  //   exit(0);
  // }

  // // Get the dataset file name from argument
  // strcpy(inputFname, argv[1]);
  // printf("Using dataset file %s\n", inputFname);

  // double *importedDataset =
  //     (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);

  // // Import data from dataset
  // int ret = ImportDataset(inputFname, importedDataset);
  // if (ret == 1) {
  //   printf("\nError importing the dataset");
  //   return 0;
  // }

  // // Check if the data parsed is correct
  // for (int i = 0; i < DIMENSION; i++) {
  //   printf("Sample Data %lf\n", importedDataset[i]);
  // }
  // *.**

  // Get the total count of dataset
  vector<int> unprocessedPoints;
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }

  printf("Preprocessed %lu data in dataset\n", unprocessedPoints.size());

  // Reset the GPU device for potential memory issues
  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));


  /**
 **************************************************************************
 * CUDA Memory allocation
 **************************************************************************
 */
  double *d_dataset;
  int *d_cluster;
  int *d_seedList;
  int *d_seedLength;
  int *d_collisionMatrix;
  int *d_extraCollision;

  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));

  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMalloc((void **)&d_seedLength, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  /**
 **************************************************************************
 * Indexing Memory allocation
 **************************************************************************
 */

 

  int *d_indexTreeMetaData;
  int *d_results;
  double *d_minPoints;
  double *d_maxPoints;
  double *d_binWidth;

  gpuErrchk(cudaMalloc((void **)&d_indexTreeMetaData,
                       sizeof(int) * TREE_LEVELS * RANGE));

  gpuErrchk(cudaMalloc((void **)&d_results,
                       sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  gpuErrchk(cudaMalloc((void **)&d_minPoints, sizeof(double) * DIMENSION));
  gpuErrchk(cudaMalloc((void **)&d_maxPoints, sizeof(double) * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_binWidth, sizeof(double) * DIMENSION));

  gpuErrchk(
      cudaMemset(d_results, -1, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  /**
 **************************************************************************
 * Assignment with default values
 **************************************************************************
 */
  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));

  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMemset(d_seedLength, 0, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  /**
  **************************************************************************
  * Initialize index structure
  **************************************************************************
  */
  // Start the time
  clock_t totalTimeStart, totalTimeStop, indexingStart, indexingStop;
  float totalTime = 0.0;

  totalTimeStart = clock();
  indexingStart = clock();

  double maxPoints[DIMENSION];
  double minPoints[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    maxPoints[j] = 0;
    minPoints[j] = 999999999;
  }

  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      if (importedDataset[i * DIMENSION + j] > maxPoints[j]) {
        maxPoints[j] = importedDataset[i * DIMENSION + j];
      }
      if (importedDataset[i * DIMENSION + j] < minPoints[j]) {
        minPoints[j] = importedDataset[i * DIMENSION + j];
      }
    }
  }

  for (int i = 0; i < DIMENSION; i++) {
    printf("Level %d Max: %f\n", i, maxPoints[i]);
    printf("Level %d Min: %f\n", i, minPoints[i]);
  }

  double binWidth[DIMENSION];
  double minBinSize = 99999999;
  for (int x = 0; x < DIMENSION; x++) {
    binWidth[x] = (double)(maxPoints[x] - minPoints[x]) / PARTITION_SIZE;
    if (minBinSize >= binWidth[x]) {
      minBinSize = binWidth[x];
    }
  }
  for (int x = 0; x < DIMENSION; x++) {
    printf("#%d Bin Width: %lf\n", x, binWidth[x]);
  }

  printf("==============================================\n");

  if (minBinSize < EPS) {
    printf("Bin width (%f) is less than EPS(%f).\n", minBinSize, EPS);
    exit(0);
  }

  // Level Partition
  int treeLevelPartition[TREE_LEVELS] = {1};

  for (int i = 0; i < DIMENSION; i++) {
    treeLevelPartition[i + 1] = PARTITION_SIZE;
  }

  int childItems[TREE_LEVELS];
  int startEndIndexes[TREE_LEVELS * RANGE];

  int mulx = 1;
  for (int k = 0; k < TREE_LEVELS; k++) {
    mulx *= treeLevelPartition[k];
    childItems[k] = mulx;
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    if (i == 0) {
      startEndIndexes[i * RANGE + 0] = 0;
      startEndIndexes[i * RANGE + 1] = 1;
      continue;
    }
    startEndIndexes[i * RANGE + 0] = startEndIndexes[((i - 1) * RANGE) + 1];
    startEndIndexes[i * RANGE + 1] = startEndIndexes[i * RANGE + 0];
    for (int k = 0; k < childItems[i - 1]; k++) {
      startEndIndexes[i * RANGE + 1] += treeLevelPartition[i];
    }
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    printf("#%d ", i);
    printf("Partition: %d ", treeLevelPartition[i]);
    printf("Range: %d %d\n", startEndIndexes[i * RANGE + 0],
           startEndIndexes[i * RANGE + 1]);
  }
  printf("==============================================\n");

  gpuErrchk(cudaMemcpy(d_minPoints, minPoints, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_maxPoints, maxPoints, sizeof(double) * DIMENSION,
  cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_binWidth, binWidth, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_indexTreeMetaData, startEndIndexes,
                       sizeof(int) * TREE_LEVELS * RANGE,
                       cudaMemcpyHostToDevice));

  int indexedStructureSize = startEndIndexes[DIMENSION * RANGE + 1];

  printf("Index Structure Size: %lf GB.\n",
         (sizeof(struct IndexStructure) * indexedStructureSize) /
             (1024 * 1024 * 1024.0));


  // Allocate memory for index buckets
  struct IndexStructure **d_indexBuckets, *d_currentIndexBucket;

  gpuErrchk(cudaMalloc((void **)&d_indexBuckets,
                       sizeof(struct IndexStructure *) * indexedStructureSize));
  
  for (int i = 0; i < indexedStructureSize; i++) {
    gpuErrchk(cudaMalloc((void **)&d_currentIndexBucket,
                         sizeof(struct IndexStructure)));
    gpuErrchk(cudaMemcpy(&d_indexBuckets[i], &d_currentIndexBucket,
                         sizeof(struct IndexStructure *),
                         cudaMemcpyHostToDevice));
  }

  // Allocate memory for current indexes stack
  int indexBucketSize = 1;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }

  indexBucketSize = indexBucketSize * THREAD_BLOCKS;

  int *d_indexesStack;

  gpuErrchk(
      cudaMalloc((void **)&d_indexesStack, sizeof(int) * indexBucketSize));

  cudaFree(d_currentIndexBucket);

  /**
 **************************************************************************
 * Data key-value pair
 **************************************************************************
 */
  int *d_dataKey;
  int *d_dataValue;
  double *d_upperBounds;

  gpuErrchk(cudaMalloc((void **)&d_dataKey, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_dataValue, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_upperBounds,
                       sizeof(double) * indexedStructureSize));


  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16*1024*1024);

  float indexingTime = 0.0;
  /**
 **************************************************************************
 * Start Indexing first
 **************************************************************************
 */
  gpuErrchk(cudaDeviceSynchronize());

  

  INDEXING_STRUCTURE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_dataset, d_indexTreeMetaData, d_minPoints, d_maxPoints, d_binWidth, d_results,
      d_indexBuckets, d_dataKey, d_dataValue, d_upperBounds);
  gpuErrchk(cudaDeviceSynchronize());
  

  /**
 **************************************************************************
 * Sorting and adjusting Data key-value pair
 **************************************************************************
 */

  thrust::sort_by_key(thrust::device, d_dataKey, d_dataKey + DATASET_COUNT,
                      d_dataValue);

  gpuErrchk(cudaDeviceSynchronize());

  INDEXING_ADJUSTMENT<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_indexTreeMetaData, d_indexBuckets, d_dataKey);

  gpuErrchk(cudaDeviceSynchronize());

  indexingStop = clock();

  printf("Index structure created.\n");

  /**
 **************************************************************************
 * Start the DBSCAN algorithm
 **************************************************************************
 */

  // Keep track of number of cluster formed without global merge
  int runningCluster = THREAD_BLOCKS;
  // Global cluster count
  int clusterCount = 0;

  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to conmtrol the while loop
  bool exit = false;

  clock_t monitorStart, monitorStop, dbscanKernelStart, dbscanKernelStop;
  float monitorTime = 0.0;
  float dbscanKernelTime = 0.0;
  float mergeTime = 0.0;
  float newSeedTime = 0.0;

  // *.**
  // NEW CODE FOR CORE SAMPLES - START
  // Vectors to track core samples during clustering
  // These will be populated during the DBSCAN process
  vector<int> coreSamples;           // Will store indices of core samples
  // vector<double> corePointCoordinates; // Will store coordinates of core samples
  // NEW CODE FOR CORE SAMPLES - END
  // *.**

  while (!exit) {

    monitorStart = clock();
    // Monitor the seed list and return the comptetion status of points
    int completed =
        MonitorSeedPoints(unprocessedPoints, &runningCluster,
                          d_cluster, d_seedList, d_seedLength,
                          d_collisionMatrix, d_extraCollision, d_results, &mergeTime, &newSeedTime);

    // printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
    //     unprocessedPoints.size());

    monitorStop = clock();
    monitorTime += (float)(monitorStop - monitorStart) / CLOCKS_PER_SEC;

    // If all points are processed, exit
    if (completed) {
      exit = true;
    }

    if (exit) break;

    dbscanKernelStart = clock();

    // *.**
    // Allocate device memory for core samples tracking
    int *d_coreSamples;
    gpuErrchk(cudaMalloc((void **)&d_coreSamples, sizeof(int) * DATASET_COUNT));
    gpuErrchk(cudaMemset(d_coreSamples, 0, sizeof(int) * DATASET_COUNT));
    // *.**

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_extraCollision, d_results, d_indexBuckets, d_indexesStack,
        d_dataValue, d_upperBounds, d_binWidth, d_minPoints, d_maxPoints,
        d_coreSamples);
    gpuErrchk(cudaDeviceSynchronize());

    dbscanKernelStop = clock();
    dbscanKernelTime += (float)(dbscanKernelStop - dbscanKernelStart) / CLOCKS_PER_SEC;
  
    // *.**
    // Get core samples directly from the DBSCAN kernel
    int* h_coreSamples = (int*)malloc(sizeof(int) * DATASET_COUNT);
    gpuErrchk(cudaMemcpy(h_coreSamples, d_coreSamples, sizeof(int) * DATASET_COUNT, cudaMemcpyDeviceToHost));
    
    // Process core samples from the device array
    for (int i = 0; i < DATASET_COUNT; i++) {
      if (h_coreSamples[i] == 1) {
        // Check if this point is already in our core samples list
        bool alreadyAdded = false;
        for (size_t j = 0; j < coreSamples.size(); j++) {
          if (coreSamples[j] == i) {
            alreadyAdded = true;
            break;
          }
        }
        
        // If not already added, add it to our core samples list
        if (!alreadyAdded) {
          coreSamples.push_back(i);
          
          // Add point coordinates to core point coordinates vector
          // for (int d = 0; d < DIMENSION; d++) {
          //   corePointCoordinates.push_back(importedDataset[i * DIMENSION + d]);
          // }
        }
      }
    }
    
    // Free temporary memory
    free(h_coreSamples);
    cudaFree(d_coreSamples);
    // *.**  
  }

  // *.**
  // NEW CODE FOR CORE SAMPLES - START
  /**
 **************************************************************************
 * Display core samples information - collected during DBSCAN process
 **************************************************************************
 */
  // Print core sample information
  printf("==============================================\n");
  printf("Core samples found: %lu\n", coreSamples.size());
  printf("==============================================\n");
  
  // Print first few core samples (if any)
  // int samplesToPrint = min(5, (int)coreSamples.size());
  // for (int i = 0; i < samplesToPrint; i++) {
  //   printf("Core sample #%d (index %d): (", i, coreSamples[i]);
  //   for (int d = 0; d < DIMENSION; d++) {
  //     printf("%lf", corePointCoordinates[i * DIMENSION + d]);
  //     if (d < DIMENSION - 1) printf(", ");
  //   }
  //   printf(")\n");
  // }
  // printf("==============================================\n");
  // NEW CODE FOR CORE SAMPLES - END
  // *.**

  /**
 **************************************************************************
 * End DBSCAN and show the results
 **************************************************************************
 */
  totalTimeStop = clock();

  printf("==============================================\n");

  printf("DBSCAN completed. Calculating clusters...\n");
  
  // *.**
  // Get the DBSCAN result
  // GetDbscanResult(d_cluster, &runningCluster, &clusterCount, &noiseCount);
  /** 
void GetDbscanResult(int *d_cluster, int *runningCluster, int *clusterCount,
                     int *noiseCount) {
  *noiseCount = thrust::count(thrust::device, d_cluster,
                              d_cluster + DATASET_COUNT, NOISE);
  int *d_localCluster;
  gpuErrchk(cudaMalloc((void **)&d_localCluster, sizeof(int) * DATASET_COUNT));
  thrust::copy(thrust::device, d_cluster, d_cluster + DATASET_COUNT,
               d_localCluster);
  thrust::sort(thrust::device, d_localCluster, d_localCluster + DATASET_COUNT);
  *clusterCount = thrust::unique(thrust::device, d_localCluster,
                                 d_localCluster + DATASET_COUNT) -
                  d_localCluster - 1;

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_localCluster,
                       sizeof(int) * DATASET_COUNT, cudaMemcpyDeviceToHost));
  ofstream outputFile;
  outputFile.open("./out/cuda_dclust_extended.txt");
  for (int j = 0; j < DATASET_COUNT; j++) {
    outputFile << localCluster[j] << endl;
  }
  outputFile.close();
  free(localCluster);

  cudaFree(d_localCluster);
}
  */
  // *.**

  totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
  indexingTime = (float)(indexingStop - indexingStart) / CLOCKS_PER_SEC;

  printf("==============================================\n");
  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);
  printf("==============================================\n");
  printf("Indexing Time: %3.2f seconds\n", indexingTime);
  printf("Merge Time: %3.2f seconds\n", mergeTime);
  printf("New Seed Fill Time: %3.2f seconds\n", newSeedTime);
  printf("DBSCAN kernel Time: %3.2f seconds\n", dbscanKernelTime);
  printf("Communication Time: %3.2f seconds\n", monitorTime - mergeTime - newSeedTime);
  printf("Total Time: %3.2f seconds\n", totalTime);
  printf("==============================================\n");

  /**
 **************************************************************************
 * Free CUDA memory allocations
 **************************************************************************
 */

  cudaFree(d_dataset);
  cudaFree(d_cluster);
  cudaFree(d_seedList);
  cudaFree(d_seedLength);
  cudaFree(d_collisionMatrix);
  cudaFree(d_extraCollision);

  // *.**
  cudaFree(d_indexTreeMetaData);
  // *.**

  cudaFree(d_results);
  // *.**
  for (int i = 0; i < indexedStructureSize; i++) { 
    cudaFree(d_indexBuckets[i]); 
  }
  // *.**
  cudaFree(d_indexBuckets);
  cudaFree(d_indexesStack);

  cudaFree(d_dataKey);
  cudaFree(d_dataValue);
  cudaFree(d_upperBounds);
  cudaFree(d_binWidth);


  cudaFree(d_minPoints);
  cudaFree(d_maxPoints);
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Import Dataset
* It imports the data from the file and store in dataset variable
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int ImportDataset(char const *fname, double *dataset) {
  FILE *fp = fopen(fname, "r");
  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }

  char buf[4096];
  unsigned long int cnt = 0;
  while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION) {
    char *field = strtok(buf, ",");
    double tmp;
    sscanf(field, "%lf", &tmp);
    dataset[cnt] = tmp;
    cnt++;

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        double tmp;
        sscanf(field, "%lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;
      }
    }
  }
  fclose(fp);
  return 0;
}