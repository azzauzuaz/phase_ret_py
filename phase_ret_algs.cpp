#include <pybind11/pybind11.h>

#include "funcs.hpp"

PYBIND11_MODULE(phase_ret_algs,m) {
    m.def("ER", &ER, "ER algorithm");
    m.def("HIO", &HIO, "HIO algorithm");
    m.def("get_error", &get_error, "Get error");
};
