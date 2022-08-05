#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <gudhi/Persistence_landscape_on_grid.h>

#include <pybind11_diagram_utils.h>

namespace py = pybind11;
namespace repr = Gudhi::Persistence_representations;

py::array_t<double> test_fct(size_t d, py::array_t<double> v){
    py::print(d);
    py::buffer_info buf1 = v.request();
    py::print(buf1.shape);
    py::array_t<double> r(d);
    return r;
}

py::array_t<double> landscape_on_grid(
        Dgm diag1, double grid_min_, double grid_max_, size_t number_of_points_,
        size_t number_of_levels_of_landscape
        ){
    py::buffer_info buf1 = diag1.request();
    size_t number_of_points_in_diag = buf1.shape[0];
    std::vector<std::pair<double, double>> diag(number_of_points_in_diag);

    double min_birth_value = std::numeric_limits<double>::max();
    double max_death_value = std::numeric_limits<double>::min();

    double *ptr = static_cast<double *>(buf1.ptr);
    for (size_t idx = 0; idx < buf1.shape[0]; idx++) {
        std::pair<double, double> pt;
        pt.first = ptr[2*idx];
        pt.second = ptr[2*idx+1];
        diag[idx] = pt;

        if (pt.first <= min_birth_value) {
            min_birth_value = pt.first;
        }
        if (pt.second >= max_death_value) {
            max_death_value = pt.second;
        }
    }

    std::vector<std::vector<double>> landscape(number_of_points_);
    if ((grid_min_ <= min_birth_value) & (max_death_value <= grid_max_)){
        landscape = repr::Persistence_landscape_on_grid(diag, grid_min_, grid_max_, number_of_points_, number_of_levels_of_landscape).landscape_on_grid();
    } else {
        repr::Persistence_landscape_on_grid landscape_on_larger_grid = repr::Persistence_landscape_on_grid(diag, min_birth_value, max_death_value, number_of_points_, number_of_levels_of_landscape);
        double dx = (grid_max_ - grid_min_)/ number_of_points_;
        for (size_t i=0; i < landscape.size(); i++) {
            std::vector<double> single_point(number_of_levels_of_landscape);
            double evaluation_point = dx*i + grid_min_;
            for (size_t j=0; j<number_of_levels_of_landscape; j++) {
                single_point[j] = landscape_on_larger_grid.compute_value_at_a_given_point(j, evaluation_point);
            }
            landscape[i] = single_point;
        }
    }

    GUDHI_CHECK(landscape.size() == number_of_points_, "Landscape dimension not of the right size");

    py::array_t<double> landscape_np = py::array_t<double>({(size_t) number_of_points_, number_of_levels_of_landscape});

    py::buffer_info buf2 = landscape_np.request();
    double *ptr2 = static_cast<double *>(buf2.ptr);

    for (size_t i = 0; i < number_of_points_; i++) {
        py::print(landscape[i]);
        for (size_t j = 0; j < landscape[i].size(); j++) {
            ptr2[i * number_of_levels_of_landscape + j] = landscape[i][j];
        }
        for (size_t j = landscape[i].size(); j < number_of_levels_of_landscape; j++) {
            ptr2[i * number_of_levels_of_landscape + j] = (double) 0.;
        }
    }
    return landscape_np;
}

PYBIND11_MODULE(landscape_cpp, m) {
    m.attr("__license__") = "GPL v3";
    m.def("landscape_on_grid", &landscape_on_grid,
    py::arg("diagram_1"), py::arg("grid_min"), py::arg("grid_max"),
    py::arg("n_points"), py::arg("n_levels")
    );
    m.def("test_fct", &test_fct, py::arg("d"), py::arg("v"));
}
