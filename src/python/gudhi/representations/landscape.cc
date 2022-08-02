#include <gudhi/Persistence_landscape_on_grid.h>

#include <pybind_diagram_utils.h>

std::vector<std::vector<double>> landscape_on_grid(
        Dgm d1, double grid_min_, double grid_max_, size_t number_of_points_,
        unsigned number_of_levels_of_landscape
        ){
    auto diag1 = numpy_to_range_of_pairs(d1);

    Gudhi::Persistence_representations::Persistence_landscape_on_grid landscape = Gudhi::Persistence_representations::Persistence_landscape_on_grid(diag1, grid_min_, grid_max_, number_of_points_, number_of_levels_of_landscape);
    return landscape.landscape_on_grid();
}

PYBIND11_MODULE(landscape, m) {
    m.attr("__license__") = "GPL v3";
    m.def("landscape_on_grid", &landscape_on_grid,
    py::arg("diagram_1"), py::arg("grid_min"), py::arg("grid_max"),
    py::arg("n_points"), py::arg("n_levels")
    );
}
