void G_DBSCAN(const float *h_data,int **clusterIDs, int **clusterType, int * numClusters,
			  int numPoints, int dataDim, int minPts, float R, int BLOCK_THREADS);