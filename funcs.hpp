#include <fftw3.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void normalize(fftw_complex *vett, double mod, int npix);

void sub_intensities(fftw_complex* vett, py::array_t<double, py::array::c_style> mod);

void apply_support_er(fftw_complex *r_space, py::array_t<double, py::array::c_style> support);

void apply_support_hio(fftw_complex *r_space, py::array_t<double, py::array::c_style> support, fftw_complex *buffer_r_space, double beta);

py::array_t<std::complex<double>, py::array::c_style> ER(py::array_t<double, py::array::c_style> intensities, py::array_t<double, py::array::c_style> support, py::array_t<std::complex<double>, py::array::c_style> r_space, int n_iterations);

py::array_t<std::complex<double>, py::array::c_style> HIO(py::array_t<double, py::array::c_style> intensities, py::array_t<double, py::array::c_style> support, py::array_t<std::complex<double>, py::array::c_style> r_space, int n_iterations, double beta);

double get_error(py::array_t<std::complex<double>, py::array::c_style> data, py::array_t<double, py::array::c_style> support, py::array_t<double, py::array::c_style> intensities);
