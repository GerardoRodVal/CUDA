from  obspy.core import read                                                        
import matplotlib.pyplot as plt
import os                                                                               

File = open('data.dat', 'r')
for j in File:
    data = map(float, j.strip().split())
    plt.plot(data)
    plt.show()

for i in os.listdir('cuda_crsmex-master'):              
    if( os.path.splitext(i)[1] == '.sac'):                                              
        Sac_file = read( i, debug_headers=True).plot()                                        
