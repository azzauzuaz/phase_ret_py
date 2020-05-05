#include "funcs.hpp"

void normalize(fftw_complex *vett, double mod, int npix){

    for(int i=0; i<npix; i++){
        vett[i][0]=vett[i][0]*1./mod;
        vett[i][1]=vett[i][1]*1./mod;
    }
};

void sub_intensities(fftw_complex* data, py::array_t<double> intensities){
    py::buffer_info int_buf = intensities.request();
    double *int_ptr = (double *) int_buf.ptr;

    for(int i=0; i<int_buf.size; i++){
        if(int_ptr[i]>-1){  //se il pixel del pattern e' noto
            double m=(data[i][1]*data[i][1]+data[i][0]*data[i][0]);
            if(m>0){
                m=sqrt(m);
                data[i][0] = int_ptr[i]*data[i][0]/(m);        // write on the input the experimental module times the real ...
                data[i][1] = int_ptr[i]*data[i][1]/(m);        // and the imaginary part of exp(i x phase)
            }
            else{
                double phase;
                if(data[i][1]<0) phase=atan2(data[i][1],data[i][0]) + 2.*M_PI;
                else phase=atan2(data[i][1],data[i][0]);
                data[i][0]=int_ptr[i]*cos(phase);
                data[i][1]=int_ptr[i]*sin(phase);
            }
        }
    }
};

void apply_support_er(fftw_complex *r_space, py::array_t<double> support){
    py::buffer_info supp_buf = support.request();
    double *supp_ptr = (double *) supp_buf.ptr;

    for(int i=0; i<supp_buf.size; i++){
        r_space[i][0]=r_space[i][0]*supp_ptr[i];
        r_space[i][1]=r_space[i][1]*supp_ptr[i];
    }
};

void apply_support_hio(fftw_complex *r_space, py::array_t<double> support, fftw_complex *buffer_r_space, double beta){
    py::buffer_info supp_buf = support.request();
    double *supp_ptr = (double *) supp_buf.ptr;

    for(int i=0; i<supp_buf.size; i++){
        r_space[i][0]=(buffer_r_space[i][0]-beta*r_space[i][0])*(1.-supp_ptr[i]) + r_space[i][0]*supp_ptr[i];
        r_space[i][1]=(buffer_r_space[i][1]-beta*r_space[i][1])*(1.-supp_ptr[i]) + r_space[i][1]*supp_ptr[i];

        buffer_r_space[i][0]=r_space[i][0];
        buffer_r_space[i][1]=r_space[i][1];
    }
};

py::array_t<std::complex<double>> ER(py::array_t<double> intensities, py::array_t<double> support, py::array_t<std::complex<double>> r_space, int n_iterations){
    py::buffer_info data_buf = r_space.request();
    std::complex<double> *data_ptr = (std::complex<double> *) data_buf.ptr;

    int x_dim=data_buf.shape[0];
    int y_dim=data_buf.shape[1];

    int npix=x_dim*y_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    //memcpy ?
    for(int i=0; i<npix; i++){
        data[i][0]=std::real(data_ptr[i]);
        data[i][1]=std::imag(data_ptr[i]);
    }

    fftw_plan p2k = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); //direttive per andare dal diretto al reciproco
    fftw_plan p2r = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i_iteration=0; i_iteration<n_iterations; i_iteration++){

        fftw_execute(p2k);                         // go in the reciprocal space
        sub_intensities(data, intensities);        // substitute experimental intensities to HIO-computed ones
        fftw_execute(p2r);                         // go back in real space
        normalize(data, npix, npix);               // normalize the obtained density (to check)
                                                   // outside HIO this is assumed, but we have to force it inside
        apply_support_er(data, support);           // see directly the comment in the function

    }

    auto output = py::array_t<std::complex<double>>(npix);
    py::buffer_info out_buf = output.request();
    std::complex<double> *out_ptr = (std::complex<double> *) out_buf.ptr;

    //memcpy ?
    for(int i=0; i<npix; i++){
        out_ptr[i].real(data[i][0]);
        out_ptr[i].imag(data[i][1]);
    }

    fftw_destroy_plan(p2k);
    fftw_destroy_plan(p2r);
    fftw_free(data);

    output.resize({x_dim,y_dim});

    return output;
};

py::array_t<std::complex<double>> HIO(py::array_t<double> intensities, py::array_t<double> support, py::array_t<std::complex<double>> r_space, int n_iterations, double beta){
    py::buffer_info data_buf = r_space.request();
    std::complex<double> *data_ptr = (std::complex<double> *) data_buf.ptr;

    int x_dim=data_buf.shape[0];
    int y_dim=data_buf.shape[1];

    int npix=x_dim*y_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);
    fftw_complex* buffer_r_space = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    //memcpy ?
    for(int i=0; i<x_dim; i++){
        data[i][0]=std::real(data_ptr[i]);
        data[i][1]=std::imag(data_ptr[i]);
        buffer_r_space[i][0]=std::real(data_ptr[i]);   // make a copy of "r_space" into "buffer_r_space"
        buffer_r_space[i][1]=std::real(data_ptr[i]);   // make a copy of "r_space" into "buffer_r_space"
    }

    fftw_plan p2k = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); //direttive per andare dal diretto al reciproco
    fftw_plan p2r = fftw_plan_dft_2d(x_dim, y_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i_iteration=0; i_iteration<n_iterations; i_iteration++){ // TO CHECK: check the possibility to use a FFT which works only with real input

        fftw_execute(p2k);                         // go in the reciprocal space
        sub_intensities(data, intensities);        // substitute experimental intensities to HIO-computed ones
        fftw_execute(p2r);                         // go back in real space
        normalize(data, npix, npix);               // normalize the obtained density (to check)

        apply_support_hio(data, support, buffer_r_space, beta);  // see directly the comment in the function
    }

    auto output = py::array_t<std::complex<double>>(npix);
    py::buffer_info out_buf = output.request();
    std::complex<double> *out_ptr = (std::complex<double> *) out_buf.ptr;

    //memcpy ?
    for(int i=0; i<npix; i++){
        out_ptr[i].real(data[i][0]);
        out_ptr[i].imag(data[i][1]);
    }

    fftw_free(buffer_r_space);
    fftw_destroy_plan(p2k);
    fftw_destroy_plan(p2r);
    fftw_free(data);

    output.resize({x_dim,y_dim});

    return output;
};

double get_error(py::array_t<std::complex<double>> data, py::array_t<double> support, py::array_t<double> intensities){
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_supp = support.request();
    py::buffer_info buf_int = intensities.request();
    std::complex<double> *ptr_data = (std::complex<double> *) buf_data.ptr;
    double *ptr_supp = (double *) buf_supp.ptr;
    double *ptr_int = (double *) buf_int.ptr;

    int x_dim=buf_data.shape[0];
    int y_dim=buf_data.shape[1];

    int npix = x_dim*y_dim;

    fftw_complex* local_data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    //memcpy ?
    for(int i=0; i<npix; i++){
        local_data[i][0]=ptr_supp[i]*std::real(ptr_data[i]);    // put densities in the real part
        local_data[i][1]=ptr_supp[i]*std::imag(ptr_data[i]);    // put zero in the imaginary part
    }

    fftw_plan p= fftw_plan_dft_2d(x_dim, y_dim, local_data, local_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    double sum=0.;
    double tot=0.;
    for(int i=0; i<npix; i++){
        if(ptr_int[i]>=0){  // no measure is available where artificially |Exp| has been set < 0
            sum += ptr_int[i];
            tot += (ptr_int[i]-sqrt(local_data[i][0]*local_data[i][0]+ local_data[i][1]*local_data[i][1]))*
                   (ptr_int[i]-sqrt(local_data[i][0]*local_data[i][0]+ local_data[i][1]*local_data[i][1]));    // tot = SUM_j | sqrt(Exp) - |FFT(real_random)| |
        }
    }


    double error=sqrt(tot)/sum;     // l'errore reale e' il rapporto tra la densita' fuori dal supporto e quella totale

    fftw_free(local_data);
    fftw_destroy_plan(p);

    return error;
};
