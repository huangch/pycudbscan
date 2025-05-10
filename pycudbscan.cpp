#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include "main.cu"  // Include the CUDA code

namespace py = pybind11;

class PyCuDBSCAN {
private:
    float eps;
    int min_samples;
    std::vector<int> labels_;
    std::vector<int> core_sample_indices_;
    float processing_time_ms;

public:
    PyCuDBSCAN(float eps = 0.5, int min_samples = 5) 
        : eps(eps), min_samples(min_samples), processing_time_ms(0.0f) {}

    py::array_t<int> fit_predict(py::array_t<float, py::array::c_style | py::array::forcecast> X) {
        // Get array info
        py::buffer_info buf = X.request();
        
        if (buf.ndim != 2) {
            throw std::runtime_error("Input array must be 2-dimensional");
        }
        
        // Get dimensions
        int n_points = buf.shape[0];
        int dims = buf.shape[1];
        
        // Convert numpy array to std::vector
        auto ptr = static_cast<float*>(buf.ptr);
        std::vector<float> points(ptr, ptr + n_points * dims);
        
        // Run DBSCAN on GPU
        labels_ = run_dbscan(points, n_points, dims, eps, min_samples, processing_time_ms);
        
        // Find core samples (this is a simplified version)
        core_sample_indices_.clear();
        for (int i = 0; i < n_points; ++i) {
            if (labels_[i] >= 0) {  // Not noise
                core_sample_indices_.push_back(i);
            }
        }
        
        // Return labels as numpy array
        py::array_t<int> result = py::array_t<int>(n_points);
        py::buffer_info result_buf = result.request();
        int* result_ptr = static_cast<int*>(result_buf.ptr);
        
        for (int i = 0; i < n_points; ++i) {
            result_ptr[i] = labels_[i];
        }
        
        return result;
    }
    
    py::array_t<int> fit(py::array_t<float, py::array::c_style | py::array::forcecast> X) {
        // Just call fit_predict and return self
        fit_predict(X);
        return py::cast(*this);
    }

    py::array_t<int> get_labels() const {
        py::array_t<int> result = py::array_t<int>(labels_.size());
        py::buffer_info buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);
        
        for (size_t i = 0; i < labels_.size(); ++i) {
            ptr[i] = labels_[i];
        }
        
        return result;
    }
    
    py::array_t<int> get_core_sample_indices() const {
        py::array_t<int> result = py::array_t<int>(core_sample_indices_.size());
        py::buffer_info buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);
        
        for (size_t i = 0; i < core_sample_indices_.size(); ++i) {
            ptr[i] = core_sample_indices_[i];
        }
        
        return result;
    }
    
    float get_processing_time_ms() const {
        return processing_time_ms;
    }
};

PYBIND11_MODULE(pycudbscan, m) {
    m.doc() = "Python bindings for GPU-accelerated DBSCAN clustering";
    
    py::class_<PyCuDBSCAN>(m, "CuDBSCAN")
        .def(py::init<float, int>(), 
             py::arg("eps") = 0.5, 
             py::arg("min_samples") = 5)
        .def("fit_predict", &PyCuDBSCAN::fit_predict, 
             "Perform DBSCAN clustering and return cluster labels")
        .def("fit", &PyCuDBSCAN::fit, 
             "Perform DBSCAN clustering")
        .def_property_readonly("labels_", &PyCuDBSCAN::get_labels, 
                              "Cluster labels for each point in the dataset")
        .def_property_readonly("core_sample_indices_", &PyCuDBSCAN::get_core_sample_indices, 
                              "Indices of core samples")
        .def_property_readonly("processing_time_ms", &PyCuDBSCAN::get_processing_time_ms, 
                              "Processing time in milliseconds");
}