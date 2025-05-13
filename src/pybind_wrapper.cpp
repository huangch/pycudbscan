#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
// Forward declaration of CUDA functions from gdbscan.cu
extern "C" {
   bool cuda_init();
   void cuda_cleanup();
   bool cuda_dbscan(
       const float* input_data,
       int n_samples,
       int n_features,
       float eps,
       int min_samples,
       int* labels,
       int* core_samples,
       int& n_clusters
   );
}
namespace py = pybind11;
class DBSCAN {
private:
   float eps;
   int min_samples;
   std::vector<int> labels_;
   std::vector<int> core_sample_indices_;  // Using vector<int> instead of vector<bool>
   int n_clusters_;
   bool fitted_;
public:
   DBSCAN(float eps = 0.5, int min_samples = 5)
       : eps(eps), min_samples(min_samples), n_clusters_(0), fitted_(false) {
       // Initialize CUDA when class is instantiated
       if (!cuda_init()) {
           throw std::runtime_error("Failed to initialize CUDA device");
       }
   }
   ~DBSCAN() {
       // Clean up CUDA resources
       cuda_cleanup();
   }
   // Main clustering method
   py::object fit(const py::array_t<float>& X) {
       auto buf = X.request();
       if (buf.ndim != 2) {
           throw std::runtime_error("Input data must be 2-dimensional");
       }
       int n_samples = buf.shape[0];
       int n_features = buf.shape[1];
       // Prepare for output data
       labels_.resize(n_samples, -1);
       core_sample_indices_.resize(n_samples, 0);  // 0 for False, 1 for True
       // Get pointer to input data
       const float* data_ptr = static_cast<const float*>(buf.ptr);
       // Call CUDA implementation
       bool success = cuda_dbscan(
           data_ptr,
           n_samples,
           n_features,
           eps,
           min_samples,
           labels_.data(),
           core_sample_indices_.data(),
           n_clusters_
       );
       if (!success) {
           throw std::runtime_error("CUDA DBSCAN clustering failed");
       }
       fitted_ = true;
       // Return self (the model instance) just like sklearn does
       return py::cast(this);
   }
   // Get cluster labels after fitting
   py::array_t<int> get_labels() const {
       if (!fitted_) {
           throw std::runtime_error("Model not yet fitted. Call fit() first.");
       }
       auto result = py::array_t<int>(labels_.size());
       py::buffer_info buf = result.request();
       int *ptr = static_cast<int *>(buf.ptr);
       std::memcpy(ptr, labels_.data(), labels_.size() * sizeof(int));
       return result;
   }
   // Get core sample boolean mask
   py::array_t<bool> get_core_sample_indices() const {
       if (!fitted_) {
           throw std::runtime_error("Model not yet fitted. Call fit() first.");
       }
       // Create a bool numpy array for output
       auto result = py::array_t<bool>(core_sample_indices_.size());
       py::buffer_info buf = result.request();
       bool *ptr = static_cast<bool *>(buf.ptr);
       // Convert int array to bool array
       for (size_t i = 0; i < core_sample_indices_.size(); i++) {
           ptr[i] = (core_sample_indices_[i] != 0);
       }
       return result;
   }
   // Get number of clusters found
   int get_n_clusters() const {
       if (!fitted_) {
           throw std::runtime_error("Model not yet fitted. Call fit() first.");
       }
       return n_clusters_;
   }
   // Fit and return predictions in one step
   py::array_t<int> fit_predict(const py::array_t<float>& X) {
       fit(X);
       return get_labels();
   }
   // Get parameters (for sklearn's get_params())
   py::dict get_params(bool deep = true) const {
       py::dict params;
       params["eps"] = eps;
       params["min_samples"] = min_samples;
       return params;
   }
   // Set parameters (for sklearn's set_params())
   void set_params(float new_eps = -1, int new_min_samples = -1) {
       if (new_eps > 0) eps = new_eps;
       if (new_min_samples > 0) min_samples = new_min_samples;
   }
};
PYBIND11_MODULE(cuda_dbscan, m) {
   m.doc() = "CUDA-accelerated DBSCAN clustering implementation compatible with scikit-learn";
   py::class_<DBSCAN>(m, "DBSCAN")
       .def(py::init<float, int>(),
            py::arg("eps") = 0.5,
            py::arg("min_samples") = 5)
       .def("fit", &DBSCAN::fit, "Fit the DBSCAN clustering model",
            py::arg("X"))
       .def("fit_predict", &DBSCAN::fit_predict,
            "Fit the model and return cluster labels",
            py::arg("X"))
       .def("get_params", &DBSCAN::get_params,
            "Get parameters for this estimator",
            py::arg("deep") = true)
       .def("set_params", &DBSCAN::set_params,
            "Set the parameters of this estimator",
            py::arg("eps") = -1,
            py::arg("min_samples") = -1)
       .def_property_readonly("labels_", &DBSCAN::get_labels,
            "Cluster labels for each point in the dataset")
       .def_property_readonly("core_sample_indices_", &DBSCAN::get_core_sample_indices,
            "Indices of core samples")
       .def_property_readonly("n_clusters_", &DBSCAN::get_n_clusters,
            "Number of clusters found")
       .def_property_readonly("_estimator_type", [](py::object) { return "clusterer"; },
            "The type of estimator");
}

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// namespace py = pybind11;

// // Forward declarations of functions defined in main.cu
// extern "C" {
//     bool check_cuda_available();
//     void cuda_dummy_function(float* input, float* output, int size);
// }

// // Wrapper function for numpy array input/output
// py::array_t<float> py_cuda_dummy_function(py::array_t<float> input) {
//     py::buffer_info buf_info = input.request();
    
//     if (buf_info.ndim != 1) {
//         throw std::runtime_error("Input must be a 1D array");
//     }
    
//     int size = buf_info.shape[0];
    
//     // Create output array of same shape
//     py::array_t<float> output(buf_info.shape);
//     py::buffer_info buf_out = output.request();
    
//     // Get raw pointers
//     float* ptr_in = static_cast<float*>(buf_info.ptr);
//     float* ptr_out = static_cast<float*>(buf_out.ptr);
    
//     // Call CUDA function
//     cuda_dummy_function(ptr_in, ptr_out, size);
    
//     return output;
// }

// // Define the Python module
// PYBIND11_MODULE(pycudbscan_core, m) {
//     m.doc() = "Python bindings for CUDA DBSCAN";
    
//     m.def("check_cuda_available", &check_cuda_available, 
//           "Check if CUDA is available");
          
//     m.def("cuda_dummy_function", &py_cuda_dummy_function, 
//           "A dummy CUDA function that multiplies each element by 2");
// }