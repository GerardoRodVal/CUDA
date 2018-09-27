from obspy import read
from numpy import array
from math import sqrt
import numpy as np
import glob

import matplotlib.pyplot as plt


def Xcorrshift( s1, s2 ):

    n1 = len(s1)
    n2 = len(s2)

    if( n1 > n2 ):
        print( "Reference signal S1 cannot be longer than target S2" )
        return 0

    nx = n2-n1+1

#---------------------------------------------- cross-correlation -------------------------------------------------------


    np.fft.fftn()

    fourier1 = np.fft.fft( s1 )                                                                                          # T. de Fourier a cada vector de cada archivo
    fourier2 = np.fft.fft( s2 )

    ComCon1 = np.conj(array(fourier1))
    ComCon2 = np.conj(array(fourier2))

    xcor = np.array( np.fft.ifft( ComCon1*ComCon2 ) ).real

# ------------------------ scale by sqrt(norm(s1)*norm(s2win)) where s2win is the moving window of s2 ------------------

    s2s2 = np.array(s2)*np.array(s2)
    scal = np.zeros(nx)                         # como se crea un vector en matlab
    scal[0] = sum(s2s2)                         # empieza en 0 o 1
    for k in range(1,nx):
        scal[k+1] = scal[k] + s2s2[n1 + k] - s2s2[k]

    scal = sqrt(scal) * np.linalg.norm(s1)
    xcor = array(xcor) / array(scal)

# ----------------------------------- optimal lag index (=delay+1) -----------------------------------------------------
    plt.subplot(3,1,1)
    plt.plot( s1 )
    plt.subplot(3, 1, 2)
    plt.plot(s2 )
    plt.subplot(3, 1, 3)
    plt.plot(xcor)
    plt.show()


for i in glob.glob('/home/gerardo/Documentos/Tesis/c_files/sac_files/*.sac'):
    for j in glob.glob('/home/gerardo/Documentos/Tesis/c_files/sac_files/*.sac'):
        if( i != j ):
            pass
            sac1 = read(i)                                                                                               # leyendo datos de archivos sac
            sac1f= sac1.filter("highpass",  freq=1.0, corners=2, zerophase=True)                                         # filtrando los datos de sac
            sac2 = read(j)
            sac2f = sac2.filter("highpass",  freq=1.0, corners=2, zerophase=True)

            s1 = sac1f[0].data                                                                                           # solamento el vector de ondas
            s2 = sac2f[0].data

            Xcorrshift( s1, s2 )
