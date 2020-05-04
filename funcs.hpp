#include <fftw3.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void normalize(fftw_complex *vett, double mod, int npix);

double myphase(double* data);

double mynorm(double* data);

void sub_intensities(fftw_complex* vett, py::array_t<double> mod, int npix);

void apply_support_er(fftw_complex *r_space, py::array_t<double> support,  int npix);

void apply_support_hio(fftw_complex *r_space, py::array_t<double> support, fftw_complex *buffer_r_space, int npix, double beta);

py::array_t<std::complex<double>> get_ft(py::array_t<std::complex<double>> input, int kx_dim, int ky_dim);

py::array_t<std::complex<double>> get_ift(py::array_t<std::complex<double>> input, int kx_dim, int ky_dim);

py::array_t<std::complex<double>> ER(py::array_t<double> intensities, py::array_t<double> support, py::array_t<std::complex<double>> r_space, int kx_dim, int ky_dim, int n_iterations);

py::array_t<std::complex<double>> HIO(py::array_t<double> intensities, py::array_t<double> support, py::array_t<std::complex<double>> r_space, int kx_dim, int ky_dim, int n_iterations, double beta);

double get_error(py::array_t<std::complex<double>> data, py::array_t<double> support, py::array_t<double> intensities, int x_dim, int y_dim);
