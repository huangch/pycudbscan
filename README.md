# PyCuDBSCAN

A Python library for GPU-accelerated DBSCAN clustering using CUDA.

## Features

- Fast DBSCAN implementation leveraging CUDA for parallel computation
- Simple Python API with scikit-learn compatible interface
- Cross-platform support (Linux, Windows, macOS)
- Optimized for large datasets

## Requirements

- CUDA Toolkit (11.0 or newer recommended)
- Python 3.7+
- A compatible NVIDIA GPU

## Installation

### Using conda

```bash
# Create a new conda environment
conda create -n pycudbscan python=3.9
conda activate pycudbscan

# Install required dependencies
conda install -c conda-forge pybind11 cmake numpy scikit-learn

# Install CUDA Toolkit (if not already installed)
conda install -c nvidia cuda-toolkit

# Clone and install the package
git clone https://github.com/yourusername/pycudbscan.git
cd pycudbscan
pip install -e .
```

### From source

```bash
git clone https://github.com/yourusername/pycudbscan.git
cd pycudbscan
pip install -e .
```

## Usage

```python
import numpy as np
from pycudbscan import CuDBSCAN

# Generate sample data
X = np.random.rand(10000, 2)

# Initialize and run DBSCAN
dbscan = CuDBSCAN(eps=0.1, min_samples=5)
labels = dbscan.fit_predict(X)

# Print number of clusters found
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters: {n_clusters}")

# Print performance metrics
print(f"Processing time: {dbscan.processing_time_ms:.2f} ms")
```

## API Reference

### `CuDBSCAN`

The main class for CUDA-accelerated DBSCAN clustering.

#### Parameters

- `eps` (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples` (int): The number of samples in a neighborhood for a point to be considered as a core point.
- `metric` (str, default='euclidean'): The distance metric to use. Currently only 'euclidean' is supported.

#### Methods

- `fit(X)`: Perform DBSCAN clustering on the input data. Returns self.
- `fit_predict(X)`: Perform DBSCAN clustering and return cluster labels. Equivalent to calling fit(X) followed by labels_.
- `get_params()`: Get parameters for this estimator.
- `set_params(**params)`: Set the parameters of this estimator.

#### Attributes

- `labels_`: Cluster labels for each point in the dataset. Noisy samples are given the label -1.
- `core_sample_indices_`: Indices of core samples.
- `components_`: Copy of each core sample found by training.
- `processing_time_ms`: Time taken for clustering in milliseconds.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.