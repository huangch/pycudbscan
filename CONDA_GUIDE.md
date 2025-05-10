# Using PyCuDBSCAN with Conda

This guide will walk you through setting up and using PyCuDBSCAN in a Conda environment.

## Setting Up the Environment

### Method 1: Using the environment.yml file

The simplest way to set up your environment is using the provided `environment.yml` file:

```bash
# Clone the repository
git clone https://github.com/yourusername/pycudbscan.git
cd pycudbscan

# Create the conda environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate pycudbscan

# The package should be installed in development mode
# You can verify with:
python -c "import pycudbscan; print(pycudbscan.__file__)"
```

### Method 2: Manual Setup

Alternatively, you can create and set up the environment manually:

```bash
# Create a new conda environment
conda create -n pycudbscan python=3.9
conda activate pycudbscan

# Install required dependencies
conda install -c conda-forge pybind11 cmake numpy scikit-learn matplotlib
conda install -c nvidia cuda-toolkit

# Install the package in development mode
pip install -e .
```

## Running the Examples

Once your environment is set up, you can run the example code:

```bash
# Make sure you're in the project directory
cd pycudbscan

# Run the example
python examples/basic_usage.py
```

This will generate a comparison between scikit-learn's DBSCAN and your GPU-accelerated DBSCAN implementation.

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your CUDA installation:
   ```bash
   nvcc --version
   ```

2. Check that your GPU is detected:
   ```bash
   nvidia-smi
   ```

3. Ensure your CUDA toolkit version is compatible with your GPU drivers.

### Build Issues

If the build fails:

1. Make sure CMake is installed and accessible:
   ```bash
   cmake --version
   ```

2. Check that you have a compatible C++ compiler:
   ```bash
   g++ --version  # Linux/macOS
   # or
   cl  # Windows (Visual Studio Command Prompt)
   ```

3. Clean and rebuild:
   ```bash
   pip uninstall -y pycudbscan
   rm -rf build/  # On Windows: rmdir /s /q build
   pip install -e .
   ```

## Using with Jupyter Notebooks

You can also use PyCuDBSCAN in Jupyter notebooks:

```bash
# Install Jupyter in your conda environment
conda install -c conda-forge jupyter

# Launch Jupyter
jupyter notebook
```

Then create a new notebook and use PyCuDBSCAN:

```python
import numpy as np
from sklearn.datasets import make_blobs
from pycudbscan import CuDBSCAN

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

# Run DBSCAN
dbscan = CuDBSCAN(eps=0.3, min_samples=10)
labels = dbscan.fit_predict(X)

print(f"Processing time: {dbscan.processing_time_ms:.2f} ms")
print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
```