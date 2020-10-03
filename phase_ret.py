import numpy as np
import phase_ret_algs as alg
from tqdm import tqdm
import cv2

def get_dim(filename):

    f = open(filename, 'r')
    fl = f.readline()
    sl = f.readline()
    w1,w2=sl.split()
    x_dim=int(w1)
    y_dim=int(w2)

    f.close()

    return x_dim, y_dim

def print_modulus(comp_data, filename, bits):

    data=np.absolute(comp_data)
    ngrey = np.power(2,bits)-1
    max=np.amax(data)
    if max==0:
        max=1

    data=np.absolute(1.*ngrey*data/max).astype(int)

    cv2.imwrite(filename, data)

def print_modulus_raw(comp_data, filename, bits):

    data=np.absolute(comp_data)
    ngrey = np.power(2,bits)-1
    head="P2\n"+str(x_dim)+" "+str(y_dim)+"\n"+str(ngrey)
    max=np.amax(data)
    if max==0:
        max=1

    data=np.absolute(1.*ngrey*data/max).astype(int)

    np.savetxt(filename, data, header=head, comments='', fmt='%i')

alg.init()

STEPS=4000
ER_ITERATIONS=20
HIO_ITERATIONS=20
SQUARE_ROOT=True
MASK=False

POISSON=True
photons=2000000

sigma=4
tau=0.04

IMPOSE_REALITY=False
HIO_BETA=0.75
R_COEFF=1

np.random.seed(1995)

print("Reading data...")

x_dim,y_dim=get_dim("INPUT/intensities.pgm")

intensities=np.loadtxt("INPUT/intensities.pgm", skiprows=3)
support=np.loadtxt("INPUT/support.pgm", skiprows=3).astype('int32')
input_data=np.loadtxt("INPUT/density.pgm", skiprows=3)

intensities=intensities.reshape((x_dim,y_dim))
support=support.reshape((x_dim,y_dim))
input_data=input_data.reshape((x_dim,y_dim))

if MASK:
    mask=np.loadtxt("INPUT/mask.pgm", skiprows=3)
    mask=mask.reshape((x_dim,y_dim))

print("Dimensions:", x_dim, "x", y_dim)

osr=support.size/np.count_nonzero(support)

print("Over-sampling ratio:",osr)

print("Setting up the pattern ...")

if POISSON:
    max_val=np.max(intensities)
    intensities=intensities/max_val*photons
    for i in range(intensities.shape[0]):
        for j in range(intensities.shape[1]):
            intensities[i][j]=np.random.poisson(intensities[i][j])

if MASK:
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]>0:
                intensities[i][j]=-1

if SQUARE_ROOT:
    for i in range(intensities.shape[0]):
        for j in range(intensities.shape[1]):
            if intensities[i][j]>=0:
                 intensities[i][j]=np.sqrt(intensities[i][j])

intensities=np.fft.fftshift(intensities)

print("Setting up the support ...")

for i in range(support.shape[0]):
    for j in range(support.shape[1]):
        if support[i][j]>0:
            support[i][j]=1

start_support=support

print("Initializing density ...")

data=input_data

data=np.fft.fft2(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        temp_phase = np.angle(data[i][j]) + R_COEFF*(np.random.rand()*2*np.pi-np.pi)
        data[i][j]=(intensities[i][j]*np.cos(temp_phase))+1j*(intensities[i][j]*np.sin(temp_phase))

data=np.fft.ifft2(data)

print_modulus_raw(data, "OUTPUT/start.pgm", 8)

error_file=open("OUTPUT/error.dat", "w", buffering=1)

print("Mainloop ...")

for i_step in tqdm(range(STEPS)):
    data=alg.HIO(intensities, support, data, HIO_ITERATIONS , HIO_BETA, IMPOSE_REALITY)
    data=alg.ER(intensities, support, data, ER_ITERATIONS, IMPOSE_REALITY)
    if i_step%10==0:
        support=alg.ShrinkWrap(data, start_support, sigma, tau)
        print_modulus_raw(support, "OUTPUT/new_support.pgm", 8)
    error=alg.get_error(data, support, intensities)
    error_file.write(str(i_step)+"   "+str(error)+"\n")
    print_modulus_raw(data, "OUTPUT/density.pgm", 8)

print_modulus_raw(data, "OUTPUT/density_final.pgm", 8)

error_file.close()

print("Final precision: ", error)
