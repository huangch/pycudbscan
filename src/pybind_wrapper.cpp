#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations of functions defined in main.cu
extern "C" {
    bool check_cuda_available();
    void cuda_dummy_function(float* input, float* output, int size);
}

// Wrapper function for numpy array input/output
py::array_t<float> py_cuda_dummy_function(py::array_t<float> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    
    int size = buf_info.shape[0];
    
    // Create output array of same shape
    py::array_t<float> output(buf_info.shape);
    py::buffer_info buf_out = output.request();
    
    // Get raw pointers
    float* ptr_in = static_cast<float*>(buf_info.ptr);
    float* ptr_out = static_cast<float*>(buf_out.ptr);
    
    // Call CUDA function
    cuda_dummy_function(ptr_in, ptr_out, size);
    
    return output;
}

// Define the Python module
PYBIND11_MODULE(pycudbscan_core, m) {
    m.doc() = "Python bindings for CUDA DBSCAN";
    
    m.def("check_cuda_available", &check_cuda_available, 
          "Check if CUDA is available");
          
    m.def("cuda_dummy_function", &py_cuda_dummy_function, 
          "A dummy CUDA function that multiplies each element by 2");
}