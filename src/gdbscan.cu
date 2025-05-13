#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "makeGraph.h"
#include "breadthFirstSearch.h"

// int NUM_NODES;
// double RADIUS;
// int MIN_POINTS;
// int BLOCK_THREADS;
// int NUM_BLOCKS;
// #define BLOCK_THREADS 256

void G_DBSCAN(const float *h_data,int **clusterIDs, int **clusterType, int * numClusters,
			  int numPoints, int dataDim, int minPts, float R, int BLOCK_THREADS){

	int NUM_BLOCKS = (numPoints/BLOCK_THREADS) +1;

	//Initialize cluster set
	for(int i=0;i<numPoints;i++){
		(*clusterIDs)[i] = Not_Visited;
		(*clusterType)[i] = Border;
	}

	Graph* distGraph = (Graph*)malloc(sizeof(Graph));

	//Create adjacency matrix and an edge between all connected nodes
	makeGraph(NUM_BLOCKS,BLOCK_THREADS, h_data, numPoints, minPts, R, distGraph, clusterType);

	//Do breadth first search to find clusters in the graph
	identifyCluster(numPoints,NUM_BLOCKS,BLOCK_THREADS, distGraph->nodes, distGraph->edges, clusterIDs, clusterType, numClusters);

	//Clean up the mess
	free(distGraph);
}


/**
 * 
 * 		defaultMin = 4;
		defaultR = 1.5;
	
		defaultPts = 400000;
 * 	  RADIUS = defaultR;
	  MIN_POINTS = defaultMin;
	  NUM_NODES = setOfDataSize[i];
		BLOCK_THREADS = 256;
		NUM_BLOCKS = (NUM_NODES/BLOCK_THREADS) +1;


 */
// void runTest(){
// 	//Initialize output of the program
// 	int *clusterIDs = (int*)malloc(sizeof(int)*NUM_NODES);
// 	bool *clusterType = (bool*)malloc(sizeof(bool)*NUM_NODES);
// 	int numClusters;

// 	//Run the G-DBScan Algorithm
// 	G_DBSCAN(h_data, &clusterIDs, &clusterType, &numClusters);
	
// 	//Clean up the mess
// 	free(clusterIDs);
// 	free(clusterType);
// }

