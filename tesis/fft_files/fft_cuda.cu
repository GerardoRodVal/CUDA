#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <cuda.h>
#include <cufft.h>
#include <math.h>
extern "C" 
{
	#include <sacio.h>
	#include <sac.h>
}

//#define DATASIZE 8
//#define BATCH 3
#define GRID_DIMENSION  3
#define BLOCK_DIMENSION 3

#define MAX 60001
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void GPUmemo( float *data, int pts )
{
	__shared__ float* trace;

	trace = (float *)malloc(pts*sizeof(float));
	int Blocks;
	for( Blocks = 0; Blocks < gridDim.x; Blocks++ )
	{
		trace[threadIdx.x] = data[threadIdx.x + Blocks*pts];
	}
}


__global__ void ComplexConj( long int nelem, cufftComplex *array )
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int thx = threadIdx.x;
	int thy = threadIdx.y;
	int thz = threadIdx.z;

	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (thx + thy*blockDim.x) + thz*(blockDim.x*blockDim.y);
	int BlockId   = (bx + by*gridDim.x) + bz*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if (uniqueid < nelem){
		array[uniqueid].y = array[uniqueid].y*-1;
 	 }
}


void ComplexGraph( cufftComplex *data, int size_fft)
{
	FILE *file;
	char filename[] = "ComplexData.dat";
	file = fopen(filename, "w");
	int l;
	for( l = 0; l<size_fft; l++ )
		fprintf(file, "%f    %f\n", data[l].x, data[l].y);

	FILE *gnuplot = NULL;
	gnuplot=popen("gnuplot","w");
	fprintf(gnuplot,"set term postscript eps enhanced color\n");
	fprintf(gnuplot, "set logscale xz\n");
	fprintf(gnuplot, "set output 'graphic_Complex.eps'\n");
	fprintf(gnuplot, "plot '%s' u 2 with lines\n", filename);
	fprintf(gnuplot, "set output\n");
	fflush(gnuplot);
	pclose(gnuplot);
	fclose(file);

}



inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s line:%d \n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



void check_gpu_card_type()
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0){
  fprintf(stderr,"ERROR - No GPU card detected.\n");
  exit(-1);
  }

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("            Device Number: %d\n", i);
    printf("              Device name: %s\n",            prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",            prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",            prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}



void Fourier( cufftComplex *fft, int batch, int size_fft )
{	
	FILE *file;
	char filename[] = "Array_.dat";
	int i;
	for( i = 0; i<batch; i++ )
	{
		filename[5] = i + '0';
		file = fopen(filename, "w");
		int j;
		for( j = 0; j < size_fft; j++ )
		{
			float result1 = fft[i*size_fft + j].x;
			float result2 = fft[i*size_fft + j].y;
			float result3 = sqrt(fft[i*size_fft + j].x*fft[i*size_fft + j].x + fft[i*size_fft + j].y*fft[i*size_fft + j].y);
			fprintf(file, "%f    %f    %f\n", result1, result2, result3 );
		}
		fclose(file);
	}
}


void Spect( int N )
{
	FILE   *gnuplot = NULL;
	char fft_file[] = "Array_.dat";
	gnuplot=popen("gnuplot","w");
	fprintf(gnuplot,"set term postscript eps enhanced color\n");
	int i;
    for( i=0; i<N; i++ )
    {
                fft_file[5] = i + '0';
                fprintf(gnuplot, "set logscale xz\n");
                fprintf(gnuplot, "set output 'graphics_fft_%i.eps'\n", i);
                fprintf(gnuplot, "plot '%s' u :(log($3)) with lines\n", fft_file);
                fprintf(gnuplot, "set output\n");
                fflush(gnuplot);

                fft_file[5] = i + '0';
                fprintf(gnuplot, "set logscale xz\n");
                fprintf(gnuplot, "set output 'graphics_%i.eps'\n", i);
                fprintf(gnuplot, "plot '%s' u :3 with lines\n", fft_file);
                fprintf(gnuplot, "set output\n");
                fflush(gnuplot);
    }
	pclose(gnuplot);
}


int main(int argc, char **argv) 
{
//----------------------------------settings to sac -----------------------------------------
  	int count = 0;
  	int nlen, nerr, max = MAX;
  	char kname[31];
	float *data;
	float yarray[MAX];
	float beg, del;

	data = (float *)malloc( 20*MAX*sizeof(float));	

	check_gpu_card_type();

// reading sac files
	struct dirent *de;  
	DIR *dr = opendir(".");								//open currently directory
    while ((de = readdir(dr)) != NULL)
    {
    	if( strstr( de->d_name, ".sac" ) ) 				// only sac files
		{
			strcpy( kname , de->d_name );				// reading sac files
		  	rsac1( kname, yarray, &nlen, &beg, &del, &max, &nerr, strlen( kname ) ) ;

			if ( nerr != 0 ) 
			{	
			    fprintf(stderr, "Error reading SAC file: %s\n", kname);
			    exit ( nerr ) ;
			}

			memcpy(&data[count*MAX], yarray, nlen*sizeof(float));
			count ++;
		}
	}

// --------------------------------------cuda_fft---------------------------------------------------
	cufftHandle plan;							// settings plan to fft
	cufftComplex *data_fft;
	cufftComplex *outfft;
	cufftReal *dev_dat;

	int rank = 1;                            
  	int n[] = { nlen };                      
  	int istride = 1, ostride = 1;            
  	int idist = MAX, odist = (nlen / 2 + 1); 
  	int inembed[] = { 0 };                   
  	int onembed[] = { 0 };                   
  	int size_fft = (nlen / 2 + 1);
  	int batch = count;   

	gpuErrchk(cudaMalloc((void**)&dev_dat, MAX*count*sizeof(cufftReal) ));
	gpuErrchk(cudaMalloc((void**)&data_fft, size_fft*count*sizeof(cufftComplex) ));
	outfft = (cufftComplex*)malloc( size_fft * count * sizeof(cufftComplex));
	gpuErrchk(cudaMemcpy(dev_dat, data, MAX*count*sizeof(float), cudaMemcpyHostToDevice));
								
	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan, dev_dat, data_fft);

//------------------------------------Complex conjugate--------------------------------------------------------
	int grid_size  = GRID_DIMENSION;
    int block_size = BLOCK_DIMENSION;

    dim3 DimGrid(grid_size, grid_size, grid_size);
    dim3 DimBlock(block_size, block_size, block_size);

    cufftHandle handle;

    cufftReal *ComCon_d;
	//cufftReal *ComCon_h;
	cufftComplex *ComCon_dO;
	cufftComplex *ComCon_hO; 
	cufftComplex *fft_conj;

	//ComCon_h = (cufftReal*)malloc(nlen*count*sizeof(cufftReal));
	ComCon_hO = (cufftComplex*)malloc((nlen) * count * sizeof(cufftComplex));
	cudaMalloc((void**)&ComCon_d, nlen*count*sizeof(cufftReal));
	cudaMalloc((void**)&ComCon_dO, (nlen) * count * sizeof(cufftComplex));
    cudaMalloc((void**)&fft_conj, (nlen) * count * sizeof(cufftComplex));

/*    for (int i=0; i<BATCH; i++)
        for (int j=0; j<DATASIZE; j++){ 
		ComCon_h[i*DATASIZE + j] = (cufftReal)((i + 1) + j);
	//	printf("ComCon_h[%d]=%f\n",i*DATASIZE + j, ComCon_h[i*DATASIZE + j]);
	}
*/
	cudaMemcpy(ComCon_d, data_fft, nlen*count*sizeof(cufftReal), cudaMemcpyDeviceToDevice);

    cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

	cufftExecR2C(handle, ComCon_d, ComCon_dO);
	cudaMemcpy(fft_conj, ComCon_dO, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

	ComplexConj<<<DimGrid,DimBlock>>>( (nlen)*count, fft_conj );

	cudaMemcpy(ComCon_hO, fft_conj, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

/*
	for (int i=0; i<BATCH; i++)
        for (int j=0; j<(DATASIZE / 2 + 1); j++)
            printf("Batch  = %i j= %i real %f imag %f\n", i, j, ComCon_hO[i*(DATASIZE / 2 + 1) + j].x, ComCon_hO[i*(DATASIZE / 2 + 1) + j].y);
*/	
	ComplexGraph( ComCon_hO, size_fft );

	cufftDestroy(handle);
    gpuErrchk(cudaFree(ComCon_dO));
    gpuErrchk(cudaFree(ComCon_d));
    gpuErrchk(cudaFree(fft_conj));

//-------------------------------------------------------------------------------------------------------------

    gpuErrchk(cudaMemcpy(outfft, data_fft, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	Fourier( outfft, batch, size_fft );
	Spect( batch );

	GPUmemo<<<count,nlen>>>( dev_dat, nlen );


	cudaFree(dev_dat);
	cudaFree(data_fft);
	cufftDestroy(plan);
	free(data);
	cudaDeviceSynchronize();
	cudaDeviceReset();
  	return (EXIT_SUCCESS);
}