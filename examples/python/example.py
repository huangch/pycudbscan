13. **examples/python/example.py** (Python example):

```python
import numpy as np
import time
from pycudbscan import check_cuda_available, cuda_dummy_function

def main():
    # Check if CUDA is available
    if not check_cuda_available():
        print("CUDA is not available. Please check your installation.")
        return
    
    print("CUDA is available")
    
    # Create sample data
    size = 1_000_000
    print(f"Creating random array with {size} elements...")
    data = np.random.rand(size).astype(np.float32)
    
    # Run the CUDA function
    print("Running CUDA dummy function...")
    start_time = time.time()
    result = cuda_dummy_function(data)
    end_time = time.time()
    
    # Verify the result
    expected = data * 2.0
    is_correct = np.allclose(result, expected)
    
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Result is correct: {is_correct}")
    
    # Show some results
    print("\nFirst 5 elements:")
    for i in range(5):
        print(f"Input: {data[i]:.6f}, Output: {result[i]:.6f}, Expected: {expected[i]:.6f}")

if __name__ == "__main__":
    main()