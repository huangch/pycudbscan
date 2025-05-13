# PyCUDBSCAN
A Python library for GPU-accelerated DBSCAN clustering using CUDA.
## Overview
PyCUDBSCAN provides a Python interface to a CUDA-based implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. This project implements a scikit-learn compatible interface to high-performance CUDA DBSCAN clustering.
By leveraging GPU parallelization, PyCUDBSCAN can process datasets orders of magnitude faster than CPU-based implementations, making it suitable for large-scale data analysis and machine learning applications.
## Features
- **CUDA Acceleration**: Harnesses the power of NVIDIA GPUs for massive parallelization
- **Scikit-learn Compatible API**: Drop-in replacement for scikit-learn's DBSCAN implementation
- **Dynamic Thread Configuration**: Automatically optimizes for different GPU architectures
- **Multi-dimensional Support**: Efficiently handles data with varying dimensions
- **Core Point Detection**: Identifies core points using GPU parallelization
## Requirements
- CUDA Toolkit (10.0+)
- C++11 compatible compiler
- Python 3.6+
- NumPy
- pybind11
- scikit-learn (for API compatibility)
## Installation
### Using pip
```bash
pip install pycudbscan
```
### From source
```bash
# Clone the repository
git clone https://github.com/huangch/pycudbscan.git
cd pycudbscan
# Install
pip install -e .
```
## Usage
```python
import numpy as np
from pycudbscan import DBSCAN
from sklearn.datasets import make_blobs
# Generate sample data
X, _ = make_blobs(n_samples=10000, centers=5, random_state=42)
X = X.astype(np.float32)  # Convert to float32 for better performance
# Create and fit the model
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan.fit(X)
# Get the results
labels = dbscan.labels_
n_clusters = dbscan.n_clusters_
core_samples = dbscan.core_sample_indices_
print(f"Found {n_clusters} clusters")
print(f"Number of core samples: {np.sum(core_samples)}")
```
## API Reference
The DBSCAN class follows scikit-learn's clustering API:
### Parameters
- **eps** (float, default=0.5): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- **min_samples** (int, default=5): The number of samples in a neighborhood for a point to be considered as a core point.
### Attributes
- **labels_**: Cluster labels for each point in the dataset. Noisy samples are given the label -1.
- **core_sample_indices_**: Boolean array indicating which points are core samples.
- **n_clusters_**: The number of clusters found.
### Methods
- **fit(X)**: Perform DBSCAN clustering from features.
- **fit_predict(X)**: Perform clustering and return cluster labels.
## Performance Comparison
On a dataset with 1 million points and 3 dimensions:
| Implementation | Time (seconds) |
|----------------|----------------|
| sklearn DBSCAN | 245.3          |
| PyCUDBSCAN     | 8.7            |
*Tested on NVIDIA RTX 3080, Intel i9-10900K*
## How It Works
PyCUDBSCAN implements the DBSCAN algorithm using CUDA to parallelize the two most computationally expensive steps:
1. **Distance Calculation**: Computing pairwise distances between points is parallelized across GPU threads
2. **Cluster Expansion**: Finding connected components is performed using parallel graph traversal
The implementation dynamically optimizes thread configuration based on the GPU architecture to maximize performance.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgements
- The CUDA implementation is based on research in GPU-accelerated DBSCAN algorithms
- Thanks to the pybind11 team for enabling seamless Python/C++ integration
- Scikit-learn for the API design
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.