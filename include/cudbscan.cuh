#ifndef CUDBSCAN_CUH
#define CUDBSCAN_CUH

#include <cuda_runtime.h>
#include <vector>

// Forward declarations of kernels
__global__ void dbscan_kernel(float* points, int n_points, int dims, 
                             float eps, int min_pts, int* labels);

// CuDBSCAN class declaration
class CuDBSCAN {
private:
    float eps;              // Epsilon parameter (neighborhood distance threshold)
    int min_pts;            // Minimum points to form a cluster
    float* d_points;        // Device pointer to data points
    int* d_labels;          // Device pointer to cluster labels
    int n_points;           // Number of data points
    int dims;               // Dimensionality of the data
    float processing_time_ms; // Processing time in milliseconds

public:
    // Constructor
    CuDBSCAN(float eps, int min_pts);
    
    // Destructor
    ~CuDBSCAN();
    
    // Run DBSCAN algorithm on the input data
    std::vector<int> fit_predict(const std::vector<float>& points, int n_points, int dims);
    
    // Get the processing time
    float get_processing_time_ms() const;
};

// Function to be called from Python
extern "C" std::vector<int> run_dbscan(const std::vector<float>& points, int n_points, int dims, 
                                     float eps, int min_pts, float& processing_time_ms);

#endif // CUDBSCAN_CUH