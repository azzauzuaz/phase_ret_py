#include "funcs.hpp"

void normalize(fftw_complex *vett, double mod, int npix){

    for(int i=0; i<npix; i++){
        vett[i][0]=vett[i][0]*1./mod;
        vett[i][1]=vett[i][1]*1./mod;
    }
};

double myphase(double* data){
    if(data[1]<0) return atan2(data[1],data[0]) + 2.*M_PI;
    else return atan2(data[1],data[0]);
};

double mynorm(double* data){
    return sqrt(data[0]*data[0]+ data[1]*data[1]);
};

void sub_intensities(fftw_complex* vett, py::array_t<double> mod, int npix){
    py::buffer_info buf = mod.request();
    double *ptr = (double *) buf.ptr;

    for(int i=0; i< npix; i++){
        if(ptr[i]>-1){  //se il pixel del pattern e' noto
            double m=(vett[i][1]*vett[i][1]+vett[i][0]*vett[i][0]);
            if(m>0){
                m=sqrt(m);
                vett[i][0] = ptr[i]*vett[i][0]/(m);        // write on the input the experimental module times the real ...
                vett[i][1] = ptr[i]*vett[i][1]/(m);        // and the imaginary part of exp(i x phase)
            }
            else{
                double phase=myphase(vett[i]);
                vett[i][0]=ptr[i]*cos(phase);
                vett[i][1]=ptr[i]*sin(phase);
            }
        }
    }

};

void apply_support_er(fftw_complex *r_space, py::array_t<double> support,  int npix){
    py::buffer_info buf = support.request();
    double *ptr = (double *) buf.ptr;

    for(int i=0; i<npix; i++){
        r_space[i][0]=r_space[i][0]*ptr[i];
        r_space[i][1]=r_space[i][1]*ptr[i];
    }
};

void apply_support_hio(fftw_complex *r_space, py::array_t<double> support, fftw_complex *buffer_r_space, int npix, double beta){
    py::buffer_info buf = support.request();
    double *ptr = (double *) buf.ptr;

    for(int i=0; i<npix; i++){

        r_space[i][0]=(buffer_r_space[i][0]-beta*r_space[i][0])*(1.-ptr[i]) + r_space[i][0]*ptr[i];
        r_space[i][1]=(buffer_r_space[i][1]-beta*r_space[i][1])*(1.-ptr[i]) + r_space[i][1]*ptr[i];

        buffer_r_space[i][0]=r_space[i][0];
        buffer_r_space[i][1]=r_space[i][1];
    }
};

py::array_t<std::complex<double>> get_ft(py::array_t<std::complex<double>> input, int kx_dim, int ky_dim) {
    py::buffer_info buf1 = input.request();

    auto output = py::array_t<std::complex<double>>(buf1.size);

    py::buffer_info buf2 = output.request();

    std::complex<double> *ptr1 = (std::complex<double> *) buf1.ptr;
    std::complex<double> *ptr2 = (std::complex<double> *) buf2.ptr;

    int npix=kx_dim*ky_dim;

    fftw_complex* temp= (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);
    fftw_plan p= fftw_plan_dft_2d(kx_dim, ky_dim, temp, temp, FFTW_BACKWARD, FFTW_ESTIMATE);

    for(int i=0; i<kx_dim*ky_dim;i++){
        temp[i][0]=std::real(ptr1[i]);
        temp[i][1]=std::imag(ptr1[i]);
    }

    fftw_execute(p);

    for(int i=0; i<kx_dim*ky_dim;i++){
        ptr2[i].real(temp[i][0]);
        ptr2[i].imag(temp[i][1]);
    }

    fftw_destroy_plan(p);
    fftw_free(temp);

    return output;
};

py::array_t<std::complex<double>> get_ift(py::array_t<std::complex<double>> input, int kx_dim, int ky_dim) {
    py::buffer_info buf1 = input.request();

    auto output = py::array_t<std::complex<double>>(buf1.size);
    py::buffer_info buf2 = output.request();

    std::complex<double> *ptr1 = (std::complex<double> *) buf1.ptr;
    std::complex<double> *ptr2 = (std::complex<double> *) buf2.ptr;

    int npix=kx_dim*ky_dim;

    fftw_complex* temp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);
    fftw_plan p= fftw_plan_dft_2d(kx_dim, ky_dim, temp, temp, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i=0; i<kx_dim*ky_dim;i++){
        temp[i][0]=std::real(ptr1[i]);
        temp[i][1]=std::imag(ptr1[i]);
    }

    fftw_execute(p);

    normalize(temp, npix,npix);

    for(int i=0; i<kx_dim*ky_dim;i++){
        ptr2[i].real(temp[i][0]);
        ptr2[i].imag(temp[i][1]);
    }

    fftw_destroy_plan(p);
    fftw_free(temp);

    return output;
};

py::array_t<std::complex<double>> ER(py::array_t<double> intensities, py::array_t<double> support, py::array_t<std::complex<double>> r_space, int kx_dim, int ky_dim, int n_iterations){
    py::buffer_info buf1 = r_space.request();

    auto output = py::array_t<std::complex<double>>(buf1.size);
    py::buffer_info buf2 = output.request();

    std::complex<double> *ptr1 = (std::complex<double> *) buf1.ptr;
    std::complex<double> *ptr2 = (std::complex<double> *) buf2.ptr;

    int npix=kx_dim*ky_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    for(int i=0; i<kx_dim*ky_dim;i++){
        data[i][0]=std::real(ptr1[i]);
        data[i][1]=std::imag(ptr1[i]);
    }

    fftw_plan p2k = fftw_plan_dft_2d(kx_dim, ky_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); //direttive per andare dal diretto al reciproco
    fftw_plan p2r = fftw_plan_dft_2d(kx_dim, ky_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i_iteration=0; i_iteration<n_iterations; i_iteration++){

        fftw_execute(p2k);                               // go in the reciprocal space
        sub_intensities(data, intensities, npix);        // substitute experimental intensities to HIO-computed ones
        fftw_execute(p2r);                               // go back in real space
        normalize(data, npix, npix);                     // normalize the obtained density (to check)
                                                         // outside HIO this is assumed, but we have to force it inside
        apply_support_er(data, support, npix);           // see directly the comment in the function

    }

    for(int i=0; i<kx_dim*ky_dim;i++){
        ptr2[i].real(data[i][0]);
        ptr2[i].imag(data[i][1]);
    }

    fftw_destroy_plan(p2k);
    fftw_destroy_plan(p2r);
    fftw_free(data);

    return output;
};

py::array_t<std::complex<double>> HIO(py::array_t<double> intensities, py::array_t<double> support, py::array_t<std::complex<double>> r_space, int kx_dim, int ky_dim, int n_iterations, double beta){
    py::buffer_info buf1 = r_space.request();

    auto output = py::array_t<std::complex<double>>(buf1.size);
    py::buffer_info buf2 = output.request();

    std::complex<double> *ptr1 = (std::complex<double> *) buf1.ptr;
    std::complex<double> *ptr2 = (std::complex<double> *) buf2.ptr;

    int npix=kx_dim*ky_dim;

    fftw_complex* data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);
    fftw_complex* buffer_r_space = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    for(int i=0; i<kx_dim*ky_dim; i++){
        data[i][0]=std::real(ptr1[i]);
        data[i][1]=std::imag(ptr1[i]);
        buffer_r_space[i][0]=std::real(ptr1[i]);    // make a copy of "r_space" into "buffer_r_space"
        buffer_r_space[i][1]=std::imag(ptr1[i]);    // note that the HIO works with the densities of the previous step
    }

    fftw_plan p2k = fftw_plan_dft_2d(kx_dim, ky_dim, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); //direttive per andare dal diretto al reciproco
    fftw_plan p2r = fftw_plan_dft_2d(kx_dim, ky_dim, data, data, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i_iteration=0; i_iteration<n_iterations; i_iteration++){ // TO CHECK: check the possibility to use a FFT which works only with real input

        fftw_execute(p2k);                               // go in the reciprocal space
        sub_intensities(data, intensities, npix);        // substitute experimental intensities to HIO-computed ones
        fftw_execute(p2r);                               // go back in real space
        normalize(data, npix, npix);                     // normalize the obtained density (to check)

        apply_support_hio(data, support, buffer_r_space, npix, beta);  // see directly the comment in the function
    }

    for(int i=0; i<kx_dim*ky_dim;i++){
        ptr2[i].real(data[i][0]);
        ptr2[i].imag(data[i][1]);
    }

    fftw_free(buffer_r_space);
    fftw_destroy_plan(p2k);
    fftw_destroy_plan(p2r);
    fftw_free(data);

    return output;
};

double get_error(py::array_t<std::complex<double>> data, py::array_t<double> support, py::array_t<double> intensities, int x_dim, int y_dim){
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_supp = support.request();
    py::buffer_info buf_int = intensities.request();
    std::complex<double> *ptr_data = (std::complex<double> *) buf_data.ptr;
    double *ptr_supp = (double *) buf_supp.ptr;
    double *ptr_int = (double *) buf_int.ptr;

    int npix = x_dim*y_dim;
    double sum=0;

    fftw_complex* local_data=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*npix);

    for(int i=0; i<npix; i++){
        local_data[i][0]=ptr_supp[i]*std::real(ptr_data[i]);    // put densities in the real part
        local_data[i][1]=ptr_supp[i]*std::imag(ptr_data[i]);    // put zero in the imaginary part
    }

    fftw_plan p= fftw_plan_dft_2d(x_dim, y_dim, local_data, local_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    double tot=0;
    for(int j=0; j<npix; j++) if(ptr_int[j]>=0){  // no measure is available where artificially |Exp| has been set < 0
        sum +=ptr_int[j];
        tot+=(ptr_int[j]-mynorm(local_data[j]))*(ptr_int[j]-mynorm(local_data[j]));    // tot = SUM_j | sqrt(Exp) - |FFT(real_random)| |
    }

    double error=sqrt(tot)/sum;     // l'errore reale e' il rapporto tra la densita' fuori dal supporto e quella totale

    fftw_free(local_data);
    fftw_destroy_plan(p);

    return error;
};
