import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from pycudbscan import CuDBSCAN
import time

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X)

# Compare with scikit-learn DBSCAN
from sklearn.cluster import DBSCAN

# Parameters
eps = 0.3
min_samples = 10

# Measure sklearn DBSCAN time
start_time = time.time()
db_sklearn = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
sklearn_time = (time.time() - start_time) * 1000  # Convert to ms
labels_sklearn = db_sklearn.labels_

# Run GPU DBSCAN
db_gpu = CuDBSCAN(eps=eps, min_samples=min_samples)
labels_gpu = db_gpu.fit_predict(X)
gpu_time = db_gpu.processing_time_ms

# Number of clusters in labels, ignoring noise if present
n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
n_clusters_gpu = len(set(labels_gpu)) - (1 if -1 in labels_gpu else 0)

print(f"Scikit-learn DBSCAN:")
print(f"  Number of clusters: {n_clusters_sklearn}")
print(f"  Processing time: {sklearn_time:.2f} ms")
print(f"\nGPU DBSCAN:")
print(f"  Number of clusters: {n_clusters_gpu}")
print(f"  Processing time: {gpu_time:.2f} ms")
print(f"\nSpeedup: {sklearn_time / gpu_time:.2f}x")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot scikit-learn DBSCAN results
colors_sklearn = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
for k, col in zip(range(n_clusters_sklearn), colors_sklearn):
    class_members = labels_sklearn == k
    axes[0].plot(X[class_members, 0], X[class_members, 1], '.', color=col, markersize=10)
axes[0].plot(X[labels_sklearn == -1, 0], X[labels_sklearn == -1, 1], '.', color='k', markersize=2)
axes[0].set_title(f'Scikit-learn DBSCAN\nClusters: {n_clusters_sklearn}, Time: {sklearn_time:.2f} ms')

# Plot GPU DBSCAN results
colors_gpu = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
for k, col in zip(range(n_clusters_gpu), colors_gpu):
    class_members = labels_gpu == k
    axes[1].plot(X[class_members, 0], X[class_members, 1], '.', color=col, markersize=10)
axes[1].plot(X[labels_gpu == -1, 0], X[labels_gpu == -1, 1], '.', color='k', markersize=2)
axes[1].set_title(f'GPU DBSCAN\nClusters: {n_clusters_gpu}, Time: {gpu_time:.2f} ms')

plt.tight_layout()
plt.savefig('dbscan_comparison.png', dpi=300)
plt.show()