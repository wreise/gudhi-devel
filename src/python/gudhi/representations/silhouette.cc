//
// Created by wreise on 05/08/22.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <gudhi/Persistence_silhouette.h>

#include <pybind11_diagram_utils.h>

namespace py = pybind11;

py::array_t<double> silhouette_on_grid(Dgm diag1, double p, double epsilon, std::pair<double, double> sample_range, size_t resolution){

    //std::pair<double, double> sample_range(sample_range_[0], sample_range_[1]);
    py::buffer_info buf1 = diag1.request();
    size_t number_of_points_in_diag = buf1.shape[0];
    std::vector<std::pair<double, double>> diag(number_of_points_in_diag);
    double *ptr = static_cast<double *>(buf1.ptr);
    for (size_t idx = 0; idx < buf1.shape[0]; idx++) {
        std::pair<double, double> pt;
        pt.first = ptr[2 * idx];
        pt.second = ptr[2 * idx + 1];
        diag[idx] = pt;
    }
    std::vector<double> silhouette = Gudhi::Persistence_representations::PersistenceSilhouette(epsilon, p, sample_range.first,
                                                                                               sample_range.second, resolution).compute_silhouette(diag);

    py::array_t<double> silhouette_np(resolution);
    py::buffer_info buf2 = silhouette_np.request();
    double *ptr2 = static_cast<double *>(buf2.ptr);
    for (size_t i=0; i<silhouette.size(); i++){
        ptr2[i] = silhouette[i];
    }
    return silhouette_np;
}

PYBIND11_MODULE(silhouette, m) {
    m.def("silhouette_cpp", &silhouette_on_grid,
          py::arg("diagram_1"), py::arg("p"), py::arg("epsilon"),
          py::arg("sample_range_"), py::arg("resolution")
    );
}