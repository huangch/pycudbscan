import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from pycudbscan import DBSCAN as CudaDBSCAN
import os
import json
from datetime import datetime

def run_comprehensive_benchmark(
    sizes=[1000, 10000, 100000],
    dimensions_list=[2, 3, 10, 50],
    eps_values=[0.1, 0.3, 0.5],
    min_samples_values=[5, 10, 20],
    n_tests=3,
    timeout=300  # Maximum time in seconds for a single test before skipping
):
    """
    Run comprehensive benchmarks comparing scikit-learn DBSCAN vs PyCUDBSCAN
    across various parameter configurations
    
    Parameters:
    -----------
    sizes : list
        List of dataset sizes to test
    dimensions_list : list
        List of dimensions to test
    eps_values : list
        DBSCAN epsilon parameters to test
    min_samples_values : list 
        DBSCAN min_samples parameters to test
    n_tests : int
        Number of times to run each test for averaging
    timeout : int
        Maximum time in seconds for a single test before skipping
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with all benchmark results
    """
    # Create results directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Prepare results DataFrame
    results = []
    
    total_tests = len(sizes) * len(dimensions_list) * len(eps_values) * len(min_samples_values)
    current_test = 0
    
    # Run benchmarks
    for size in sizes:
        for dimensions in dimensions_list:
            print(f"\n{'='*80}")
            print(f"Benchmarking with {size} points, {dimensions} dimensions...")
            print(f"{'='*80}")
            
            # Generate dataset - reuse for all eps and min_samples with this size/dimension
            print(f"Generating dataset...", end='', flush=True)
            centers = min(10, size//1000 + 3)  # Scale centers with data size
            X, _ = make_blobs(n_samples=size, 
                              centers=centers,
                              n_features=dimensions, 
                              random_state=42)
            X = X.astype(np.float32)  # Convert to float32 for CUDA
            print(f" done. Created dataset with {size} points, {dimensions} dimensions, {centers} centers.")
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    current_test += 1
                    print(f"\nTest {current_test}/{total_tests}: size={size}, dimensions={dimensions}, " +
                          f"eps={eps}, min_samples={min_samples}")
                    
                    # Arrays to store times for averaging
                    sklearn_run_times = []
                    cuda_run_times = []
                    sklearn_clusters = None
                    cuda_clusters = None
                    
                    # Try scikit-learn implementation
                    sklearn_status = "completed"
                    for i in range(n_tests):
                        print(f"  scikit-learn DBSCAN run {i+1}/{n_tests}...", end='', flush=True)
                        sklearn_model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                        start_time = time.time()
                        try:
                            # Add timeout check
                            finished = False
                            sklearn_model.fit(X)
                            end_time = time.time()
                            sklearn_time = end_time - start_time
                            
                            if sklearn_time > timeout:
                                print(f" exceeded timeout ({sklearn_time:.2f}s > {timeout}s)")
                                sklearn_status = "timeout"
                                break
                            
                            sklearn_run_times.append(sklearn_time)
                            print(f" done in {sklearn_time:.4f}s")
                            
                            # Get number of clusters on first successful run
                            if i == 0:
                                labels = sklearn_model.labels_
                                sklearn_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                noise_points = np.sum(labels == -1)
                                print(f"    Found {sklearn_clusters} clusters, {noise_points} noise points")
                                
                        except Exception as e:
                            print(f" failed with error: {e}")
                            sklearn_status = "error"
                            break
                    
                    # Try CUDA implementation
                    cuda_status = "completed"
                    for i in range(n_tests):
                        print(f"  PyCUDBSCAN run {i+1}/{n_tests}...", end='', flush=True)
                        cuda_model = CudaDBSCAN(eps=eps, min_samples=min_samples)
                        start_time = time.time()
                        try:
                            cuda_model.fit(X)
                            end_time = time.time()
                            cuda_time = end_time - start_time
                            
                            if cuda_time > timeout:
                                print(f" exceeded timeout ({cuda_time:.2f}s > {timeout}s)")
                                cuda_status = "timeout"
                                break
                                
                            cuda_run_times.append(cuda_time)
                            print(f" done in {cuda_time:.4f}s")
                            
                            # Get number of clusters on first successful run
                            if i == 0:
                                labels = cuda_model.labels_
                                cuda_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                noise_points = np.sum(labels == -1)
                                print(f"    Found {cuda_clusters} clusters, {noise_points} noise points")
                                
                        except Exception as e:
                            print(f" failed with error: {e}")
                            cuda_status = "error"
                            break
                    
                    # Calculate average times and speedup
                    avg_sklearn_time = np.mean(sklearn_run_times) if sklearn_run_times else None
                    avg_cuda_time = np.mean(cuda_run_times) if cuda_run_times else None
                    
                    if avg_sklearn_time and avg_cuda_time:
                        speedup = avg_sklearn_time / avg_cuda_time
                    else:
                        speedup = None
                    
                    # Store results
                    result = {
                        'size': size,
                        'dimensions': dimensions,
                        'eps': eps,
                        'min_samples': min_samples,
                        'sklearn_time': avg_sklearn_time,
                        'cuda_time': avg_cuda_time,
                        'speedup': speedup,
                        'sklearn_clusters': sklearn_clusters,
                        'cuda_clusters': cuda_clusters,
                        'sklearn_status': sklearn_status,
                        'cuda_status': cuda_status
                    }
                    results.append(result)
                    
                    # Print summary of this test
                    print(f"\n  Results:")
                    print(f"    scikit-learn: {avg_sklearn_time:.4f}s ({sklearn_status})")
                    print(f"    PyCUDBSCAN: {avg_cuda_time:.4f}s ({cuda_status})")
                    if speedup:
                        print(f"    Speedup: {speedup:.2f}x")
                    
                    # Save intermediate results
                    results_df = pd.DataFrame(results)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_df.to_csv(f"benchmark_results/dbscan_results_{timestamp}.csv", index=False)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    return results_df

def plot_comprehensive_results(results_df):
    """Create visualizations for the comprehensive benchmark results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("benchmark_plots", exist_ok=True)
    
    # 1. Speedup by dataset size and dimensions
    plt.figure(figsize=(12, 8))
    completed_df = results_df[(results_df['sklearn_status'] == 'completed') & 
                              (results_df['cuda_status'] == 'completed')]
    
    if len(completed_df) > 0:
        # Prepare data for pivot table
        pivot_data = completed_df.pivot_table(
            index='size', 
            columns='dimensions',
            values='speedup',
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title('Average Speedup (scikit-learn / PyCUDBSCAN)')
        plt.xlabel('Dimensions')
        plt.ylabel('Dataset Size')
        plt.savefig(f"benchmark_plots/speedup_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # 2. Execution time by implementation and dataset size
    plt.figure(figsize=(14, 8))
    
    for dim in completed_df['dimensions'].unique():
        dim_df = completed_df[completed_df['dimensions'] == dim]
        
        if len(dim_df) > 0:
            plt.figure(figsize=(14, 8))
            
            # Plot both implementations
            plt.plot(dim_df['size'], dim_df['sklearn_time'], 'o-', 
                     label=f'scikit-learn (dim={dim})')
            plt.plot(dim_df['size'], dim_df['cuda_time'], 's-', 
                     label=f'PyCUDBSCAN (dim={dim})')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.xlabel('Dataset Size')
            plt.ylabel('Execution Time (seconds)')
            plt.title(f'DBSCAN Performance Comparison - {dim} Dimensions')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"benchmark_plots/time_comparison_dim{dim}_{timestamp}.png", dpi=300)
            plt.close()
    
    # 3. Parameter influence on speedup
    plt.figure(figsize=(15, 10))
    
    # Influence of eps
    plt.subplot(2, 1, 1)
    sns.boxplot(x='eps', y='speedup', data=completed_df)
    plt.title('Influence of epsilon parameter on speedup')
    plt.xlabel('Epsilon')
    plt.ylabel('Speedup')
    plt.grid(True, axis='y')
    
    # Influence of min_samples
    plt.subplot(2, 1, 2)
    sns.boxplot(x='min_samples', y='speedup', data=completed_df)
    plt.title('Influence of min_samples parameter on speedup')
    plt.xlabel('Min Samples')
    plt.ylabel('Speedup')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"benchmark_plots/parameter_influence_{timestamp}.png", dpi=300)
    
    # 4. Clustering agreement
    if 'sklearn_clusters' in completed_df.columns and 'cuda_clusters' in completed_df.columns:
        completed_df['clusters_match'] = completed_df['sklearn_clusters'] == completed_df['cuda_clusters']
        agreement_pct = (completed_df['clusters_match'].sum() / len(completed_df)) * 100
        
        plt.figure(figsize=(10, 6))
        agreement_df = completed_df.groupby(['dimensions', 'eps'])['clusters_match'].mean().reset_index()
        pivot = agreement_df.pivot_table(index='dimensions', columns='eps', values='clusters_match')
        
        sns.heatmap(pivot, annot=True, cmap="RdYlGn", vmin=0, vmax=1, fmt=".2f")
        plt.title(f'Clustering Agreement Rate: {agreement_pct:.1f}%\n(sklearn clusters = cuda clusters)')
        plt.xlabel('Epsilon')
        plt.ylabel('Dimensions')
        plt.savefig(f"benchmark_plots/clustering_agreement_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # 5. Success rate analysis
    plt.figure(figsize=(12, 6))
    
    # Count status combinations
    status_counts = results_df.groupby(['sklearn_status', 'cuda_status']).size().reset_index()
    status_counts.columns = ['sklearn_status', 'cuda_status', 'count']
    
    # Create a pivot table for the heatmap
    status_pivot = status_counts.pivot_table(
        index='sklearn_status', 
        columns='cuda_status', 
        values='count',
        fill_value=0
    )
    
    sns.heatmap(status_pivot, annot=True, fmt="d", cmap="Blues")
    plt.title('Test Completion Status')
    plt.xlabel('PyCUDBSCAN Status')
    plt.ylabel('scikit-learn Status')
    plt.savefig(f"benchmark_plots/completion_status_{timestamp}.png", dpi=300, bbox_inches='tight')

def print_system_info():
    """Print system information for reproducibility"""
    import platform
    import scipy
    import sklearn
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
            cuda_capability = torch.cuda.get_device_capability(0)
        else:
            cuda_device = "N/A"
            cuda_capability = "N/A"
    except ImportError:
        cuda_available = "Unknown (torch not installed)"
        cuda_device = "Unknown"
        cuda_capability = "Unknown"
    
    system_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "processor": platform.processor(),
        "cuda_available": str(cuda_available),
        "cuda_device": cuda_device,
        "cuda_capability": str(cuda_capability)
    }
    
    try:
        from pycudbscan import __version__ as pycudbscan_version
        system_info["pycudbscan_version"] = pycudbscan_version
    except (ImportError, AttributeError):
        system_info["pycudbscan_version"] = "Unknown"
    
    # Print information
    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    # Save system info to file
    os.makedirs("benchmark_results", exist_ok=True)
    with open("benchmark_results/system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    
    return system_info

def main():
    print("Comprehensive DBSCAN Benchmark: scikit-learn vs PyCUDBSCAN\n")
    system_info = print_system_info()
    
    # Define test parameters - ADJUST THESE FOR YOUR BENCHMARK NEEDS
    
    # Quick test (fast)
    sizes = [1000, 5000, 10000]
    dimensions_list = [2, 3, 10]
    eps_values = [0.3]
    min_samples_values = [10]
    n_tests = 2
    timeout = 60  # 1 minute timeout
    
    # Medium test (moderate run time)
    # sizes = [1000, 5000, 10000, 50000]
    # dimensions_list = [2, 3, 10, 25]
    # eps_values = [0.2, 0.3, 0.5]
    # min_samples_values = [5, 10, 20]
    # n_tests = 3
    # timeout = 180  # 3 minute timeout
    
    # Comprehensive test (long run time)
    # sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    # dimensions_list = [2, 3, 10, 25, 50, 100]
    # eps_values = [0.1, 0.2, 0.3, 0.5, 0.7]
    # min_samples_values = [5, 10, 20, 50]
    # n_tests = 3
    # timeout = 300  # 5 minute timeout
    
    # Run benchmarks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nStarting benchmark at {timestamp}")
    print(f"Configuration: {len(sizes)} sizes × {len(dimensions_list)} dimensions × " +
          f"{len(eps_values)} eps values × {len(min_samples_values)} min_samples values " +
          f"× {n_tests} repetitions")
    print(f"That's {len(sizes) * len(dimensions_list) * len(eps_values) * len(min_samples_values)} unique configurations")
    
    try:
        results_df = run_comprehensive_benchmark(
            sizes=sizes,
            dimensions_list=dimensions_list,
            eps_values=eps_values,
            min_samples_values=min_samples_values,
            n_tests=n_tests,
            timeout=timeout
        )
        
        # Save final results
        final_path = f"benchmark_results/dbscan_final_results_{timestamp}.csv"
        results_df.to_csv(final_path, index=False)
        print(f"\nFinal results saved to {final_path}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        plot_comprehensive_results(results_df)
        print("Visualizations saved to benchmark_plots/ directory")
        
        # Print summary statistics
        print("\nBenchmark Summary:")
        completed_df = results_df[(results_df['sklearn_status'] == 'completed') & 
                                 (results_df['cuda_status'] == 'completed')]
        
        if len(completed_df) > 0:
            avg_speedup = completed_df['speedup'].mean()
            max_speedup = completed_df['speedup'].max()
            min_speedup = completed_df['speedup'].min()
            
            print(f"Tests completed successfully: {len(completed_df)} out of {len(results_df)}")
            print(f"Average speedup: {avg_speedup:.2f}x")
            print(f"Maximum speedup: {max_speedup:.2f}x")
            print(f"Minimum speedup: {min_speedup:.2f}x")
            
            # Show average speedup by dimension
            print("\nAverage speedup by dimension:")
            dim_speedup = completed_df.groupby('dimensions')['speedup'].mean()
            for dim, speedup in dim_speedup.items():
                print(f"  {dim} dimensions: {speedup:.2f}x")
            
            # Show average speedup by dataset size
            print("\nAverage speedup by dataset size:")
            size_speedup = completed_df.groupby('size')['speedup'].mean()
            for size, speedup in size_speedup.items():
                print(f"  {size} points: {speedup:.2f}x")
        else:
            print("No tests completed successfully for both implementations.")
            
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()