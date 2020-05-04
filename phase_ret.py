#!/usr/local/bin/python3

import numpy as np
import phase_ret_algs as alg
from tqdm import tqdm

def get_dim(filename):

    f = open(filename, 'r')
    fl = f.readline()
    sl = f.readline()
    w1,w2=sl.split()
    x_dim=int(w1)
    y_dim=int(w2)

    f.close()

    return x_dim, y_dim

def shift(data, x_trasl, y_trasl, x_dim, y_dim):
    npix=x_dim*y_dim

    x_trasl*=-1
    y_trasl*=-1
    data=alg.get_ft(data, x_dim, y_dim)
    for i in range(npix):
        kx=i%x_dim
        ky=i/y_dim

        shift=2*np.pi*(x_trasl*kx/x_dim+y_trasl*ky/y_dim)
        norm=np.absolute(data[i])
        phase=np.angle(data[i])

        data[i]=(norm+np.cos(phase+shift))+1j*(norm+np.sin(phase+shift))

    data=alg.get_ift(data, x_dim, y_dim)

def print_modulus(comp_data, x_dim, y_dim, filename, bits):
    npix=x_dim*y_dim

    data=np.zeros(npix)
    for i in range(npix):
        data[i]=np.absolute(comp_data[i])

    f= open(filename,"w")

    ngrey = np.power(2,bits)-1
    f.write("P2\n"+str(x_dim)+" "+str(y_dim)+"\n"+str(ngrey)+"\n")
    max=0
    for i in range(npix):
        if data[i]>max:
            max=data[i]
    if max==0:
        max=1
    for i in range(npix):
        temp = np.absolute( 1.*ngrey*data[i]/max).astype(int)
        f.write(str(temp)+"\n")

    f.close()

SQUARE_ROOT=1
STEPS=50
ITERATIONS=20

HIO_BETA=0.75
R_COEFF=1

print("Reading data...")

x_dim,y_dim=get_dim("INPUT/intensities.raw")

intensities=np.loadtxt("INPUT/intensities.raw", skiprows=3)
support=np.loadtxt("INPUT/support.raw", skiprows=3)
input_data=np.loadtxt("INPUT/density.raw", skiprows=3)

intensities=intensities.flatten()

npix=x_dim*y_dim

print("Dimensions: "+str(x_dim)+" x "+str(y_dim))

reg_density=0
for i in range(npix):
    if support[i]!=0:
        reg_density+=1

sigma=npix/reg_density

print("Over-sampling ratio: "+str(sigma))

print("Setting up the pattern ...")
x_center=x_dim/2
y_center=y_dim/2;

if SQUARE_ROOT:
    for i in range(npix):
        if intensities[i]>=0:
            intensities[i]=np.sqrt(intensities[i])

shift(intensities, x_center, y_center, x_dim, y_dim)

print("Setting up the support ...")

for i in range(npix):
    if support[i]>0:
        support[i]=1

print("Initializing density ...")

data=input_data

data=alg.get_ft(data, x_dim, y_dim)

for i in range(npix):
    temp_phase = np.angle(data[i]) + R_COEFF*(np.random.rand()*2*np.pi-np.pi)
    data[i]=(intensities[i]*np.cos(temp_phase))+1j*(intensities[i]*np.sin(temp_phase))

data=alg.get_ift(data, x_dim, y_dim)

print_modulus(data, x_dim, y_dim, "OUTPUT/start.pgm", 8)

error_file=open("OUTPUT/error.dat", "w")

print("Mainloop ...")

for i_step in tqdm(range(STEPS)):
    data=alg.ER(intensities, support, data, x_dim, y_dim, ITERATIONS)
    data=alg.HIO(intensities, support, data, x_dim, y_dim, ITERATIONS , HIO_BETA)
    error=alg.get_error(data, support, intensities, x_dim, y_dim)
    error_file.write(str(i_step)+"   "+str(error)+"\n")
    print_modulus(data, x_dim, y_dim, "OUTPUT/density.pgm", 8)

print_modulus(data, x_dim, y_dim, "OUTPUT/density_final.pgm", 8)

error_file.close()

print("Final precision: "+str(error))
