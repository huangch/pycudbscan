import unittest
import numpy as np
from pycudbscan import CuDBSCAN

class TestCuDBSCAN(unittest.TestCase):
    
    def setUp(self):
        # Create simple dataset with 3 distinct clusters
        np.random.seed(42)
        
        # Cluster 1
        self.cluster1 = np.random.randn(100, 2) * 0.3 + np.array([2, 2])
        
        # Cluster 2
        self.cluster2 = np.random.randn(100, 2) * 0.3 + np.array([-2, -2])
        
        # Cluster 3
        self.cluster3 = np.random.randn(100, 2) * 0.3 + np.array([2, -2])
        
        # Combine all points
        self.X = np.vstack([self.cluster1, self.cluster2, self.cluster3])
        
    def test_basic_clustering(self):
        # Parameters for clear separation
        dbscan = CuDBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(self.X)
        
        # We should have 3 clusters (ignoring noise points)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.assertEqual(num_clusters, 3, "Should identify 3 clusters")
        
        # Each distinct spatial cluster should have the same label
        # (but we don't know which label will be assigned to which cluster)
        unique_labels = set(labels)
        
        # Remove noise label if present
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        self.assertLessEqual(len(unique_labels), 3, "Should not have more than 3 clusters")
        
    def test_noise_handling(self):
        # Add some noise points
        noise_points = np.random.uniform(-5, 5, (20, 2))
        X_with_noise = np.vstack([self.X, noise_points])
        
        # Parameters that should identify the noise
        dbscan = CuDBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_with_noise)
        
        # Count noise points (label -1)
        noise_count = np.sum(labels == -1)
        self.assertGreater(noise_count, 0, "Should identify some noise points")
        
    def test_processing_time(self):
        # Make sure the processing time is recorded
        dbscan = CuDBSCAN(eps=0.5, min_samples=5)
        dbscan.fit_predict(self.X)
        
        self.assertGreater(dbscan.processing_time_ms, 0, 
                         "Processing time should be greater than 0")

if __name__ == '__main__':
    unittest.main()