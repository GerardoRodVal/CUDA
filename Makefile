pruebaCuda.o: pruebaCuda.cu
	nvcc pruebaCuda.cu -o pruebaCuda.o -I. -I/usr/local/sac/include -L/usr/local/sac/lib -lsac -lsacio -lm -lcufft
	
# pruebaCuda.o: pruebaCuda.cu
#	nvcc pruebaCuda.cu -o pruebaCuda.o -I. -I/usr/local/sac/include -L/usr/local/sac/lib -lsac -lsacio -lm
