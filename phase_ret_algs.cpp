#include <pybind11/pybind11.h>

#include "funcs.hpp"

namespace py = pybind11;

PYBIND11_MODULE(phase_ret_algs,m) {
    m.def("get_ft", &get_ft, "Get Fourier Transform!");
    m.def("get_ift", &get_ift, "Get Inverse Fourier Transform!");
    m.def("ER", &ER, "ER algorithm");
    m.def("HIO", &HIO, "HIO algorithm");
    m.def("get_error", &get_error, "Error Calculation");
};
