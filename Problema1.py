from obspy import read
from numpy import array
from math import sqrt
import numpy as np
import glob

import matplotlib.pyplot as plt


def nextPowerOf2(n):

    count = 0;
    if (n and not (n & (n - 1))):
        return n

    while (n != 0):
        n >>= 1
        count += 1

    return 1 << count;


def Xcorrshift( s1, s2 ):

    n1 = len(s1)
    n2 = len(s2)

    if( n1 > n2 ):
        print( "Reference signal S1 cannot be longer than target S2" )
        return 0

    nx = n2-n1 +1

    win = range(nx)
    only_pos = 0

#---------------------------------------------- cross-correlation -------------------------------------------------------

    nfft = (nextPowerOf2( n2+n1 ))

    fourier1 = np.fft.fft( s1, nfft )
    fourier2 = np.fft.fft( s2, nfft )

    f1 = array(np.conj(fourier1))
    f2 = array(fourier2)

    xcor = np.array( np.fft.ifft( f1*f2 ) ).real

    xcor = xcor[:nx]

# ------------------------ scale by sqrt(norm(s1)*norm(s2win)) where s2win is the moving window of s2 ------------------

    s2s2 = np.array(s2)*np.array(s2)
    scal = np.zeros(nx)
    scal[0] = sum(s2s2[:n1])

    scal = list(scal)

    for k in range(nx-1):
        scal[k+1] = scal[k] + s2s2[n1 + k] - s2s2[k]

    scal = np.sqrt(scal) * array( np.linalg.norm(s1) )

    xcor = array(xcor) / array(scal)

# ----------------------------------- optimal lag index (=delay+1) -----------------------------------------------------

    xcor = list(xcor)
    if( only_pos == 0 ):
        maxc = max( xcor )
        maxi = xcor.index( maxc )
    else:
        maxc = max(abs(xcor))
        maxi = xcor.index( maxc )

    maxi = maxi + win[1]-1
    maxc = xcor[maxi]

# -------------------------------------------------- sub-sample precision ----------------------------------------------

    if( maxi > 1 and maxi < nx-1 ):
        xc1 = (0.5)*(xcor[maxi+1]-xcor[maxi-1])
        xc2 = (xcor[maxi-1]) - (2*xcor[maxi]+xcor[maxi+1] )
        maxi = maxi-xc1/xc2
        maxc = maxc - 0.5*xc1*xc1/xc2

# -------------------------------------------------- lag --> delay -----------------------------------------------------

    dsamp = maxi-1
    
# ----------------------------------------------------- plot -----------------------------------------------------------
    plt.subplot(4,1,1)
    plt.plot( s1, 'k' )
    plt.subplot(4, 1, 2)
    plt.plot(s2 )
    plt.subplot(4, 1, 3)
    plt.plot(xcor, 'g')

    plt.subplot(4, 1, 4)
    plt.plot(s1, 'k')
    plt.subplot(4, 1, 4)
    plt.plot(s2)

    plt.show()



for i in glob.glob('/home/gerardo/Documentos/Tesis/c_files/sac_files/*.sac'):
    for j in glob.glob('/home/gerardo/Documentos/Tesis/c_files/sac_files/*.sac'):
        if( i != j ):
            pass
            sac1 = read(i)                                                                                               # leyendo datos de archivos sac
            sac1f= sac1.filter("highpass",  freq=1.0, corners=2, zerophase=True)                                         # filtrando los datos de sac
            sac2 = read(j)
            sac2f = sac2.filter("highpass",  freq=1.0, corners=2, zerophase=True)

            s1 = sac1f[0].data[:50000]                                                                                        # solamento el vector de ondas
            s2 = sac2f[0].data

            Xcorrshift( s1, s2 )
