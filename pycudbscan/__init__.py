"""
PyCUDBSCAN: GPU-accelerated DBSCAN clustering using CUDA.

This package provides a scikit-learn compatible implementation of DBSCAN
(Density-Based Spatial Clustering of Applications with Noise) that runs on 
NVIDIA GPUs for significantly improved performance.
"""

__version__ = '0.1.0'

# Import the DBSCAN class from the compiled CUDA module
from .cuda_dbscan import DBSCAN

# Define what should be available when someone does "from pycudbscan import *"
__all__ = ['DBSCAN']


def cuda_available():
    """
    Check if CUDA is available and accessible by the package.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        # Create a minimal DBSCAN instance to test CUDA initialization
        DBSCAN()
        return True
    except RuntimeError:
        # CUDA initialization failed
        return False


def compare_with_sklearn(X, eps=0.5, min_samples=5, print_results=True):
    """
    Compare performance between PyCUDBSCAN and scikit-learn's DBSCAN.
    
    Parameters:
        X (numpy.ndarray): Input data for clustering
        eps (float): The maximum distance between two samples for one to be 
                    considered as in the neighborhood of the other
        min_samples (int): The number of samples in a neighborhood for a point 
                           to be considered as a core point
        print_results (bool): Whether to print the results
        
    Returns:
        dict: Dictionary containing performance metrics and comparison results
    """
    try:
        import time
        import numpy as np
        from sklearn.cluster import DBSCAN as SklearnDBSCAN
        
        # Convert to float32 for CUDA processing
        X_float32 = X.astype(np.float32)
        
        # Benchmark scikit-learn
        start_time = time.time()
        sklearn_model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        sklearn_model.fit(X_float32)
        sklearn_time = time.time() - start_time
        
        # Benchmark PyCUDBSCAN
        start_time = time.time()
        cuda_model = DBSCAN(eps=eps, min_samples=min_samples)
        cuda_model.fit(X_float32)
        cuda_time = time.time() - start_time
        
        # Calculate metrics
        speedup = sklearn_time / cuda_time if cuda_time > 0 else float('inf')
        
        # Compare clustering results
        sklearn_clusters = len(set(sklearn_model.labels_)) - (1 if -1 in sklearn_model.labels_ else 0)
        cuda_clusters = cuda_model.n_clusters_
        
        # Check if the clusters match
        clusters_match = sklearn_clusters == cuda_clusters
        
        # Compute agreement percentage
        agreement = np.mean(sklearn_model.labels_ == cuda_model.labels_) * 100 if len(sklearn_model.labels_) > 0 else 0
        
        results = {
            "sklearn_time": sklearn_time,
            "cuda_time": cuda_time,
            "speedup": speedup,
            "sklearn_clusters": sklearn_clusters,
            "cuda_clusters": cuda_clusters,
            "clusters_match": clusters_match,
            "agreement_percentage": agreement
        }
        
        if print_results:
            print(f"Performance Comparison:")
            print(f"  scikit-learn DBSCAN: {sklearn_time:.4f}s")
            print(f"  PyCUDBSCAN: {cuda_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"\nClustering Results:")
            print(f"  scikit-learn clusters: {sklearn_clusters}")
            print(f"  PyCUDBSCAN clusters: {cuda_clusters}")
            print(f"  Clusters match: {clusters_match}")
            print(f"  Labels agreement: {agreement:.2f}%")
        
        return results
        
    except ImportError:
        if print_results:
            print("scikit-learn not available for comparison")
        return {"error": "scikit-learn not available"}
    except Exception as e:
        if print_results:
            print(f"Error during comparison: {str(e)}")
        return {"error": str(e)}
    
# from .pycudbscan_core import check_cuda_available, cuda_dummy_function

# __version__ = '0.1.0'

# __all__ = [
#     'check_cuda_available',
#     'cuda_dummy_function',
# ]