#include <algorithm>
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

#define Nsac       		2
#define GRID_DIMENSION  3
#define BLOCK_DIMENSION 3
#define MAX 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


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

__global__ void Correlate( cufftComplex *Input, cufftComplex *Output, int xlen )
{
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	/*
	if( uniqueid < xlen ){
		Output[uniqueid].x = Input[uniqueid].x;
		Output[uniqueid].y = Input[uniqueid].y;
	}*/
}

__global__ void VectorMult( cufftComplex *Input, cufftComplex *Output, int xlen, int npts )
{
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if( uniqueid < xlen ){
		Output[uniqueid].x = (Input[uniqueid].x * Input[uniqueid].x + Input[uniqueid].y * Input[uniqueid].y)/npts;
		Output[uniqueid].y = 0;
	}
}

int main(int argc, char **argv) 
{
//---------------------------------------------Time event-------------------------------------------------------
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
//------------------------------------------ Kernel invocation--------------------------------------------------
	int grid_size  = GRID_DIMENSION;
    int block_size = BLOCK_DIMENSION;

    dim3 DimGrid(grid_size, grid_size, grid_size);
    dim3 DimBlock(block_size, block_size, block_size);

//--------------------------------------------settings to sac --------------------------------------------------
  	int count = 0;
  	int nlen, nerr, max = MAX;
  	char kname[31];
	float *data;
	float yarray[MAX];
	float beg, del;

	data = (float *)malloc( Nsac*MAX*sizeof(float));

	check_gpu_card_type();

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

// ---------------------------------------------fft_settings---------------------------------------------------
	int DATASIZE = MAX;
	int size_fft = DATASIZE / 2 + 1;
  	int batch    = count;    

// -----------------------------------------------cuda_fft-----------------------------------------------------
	cufftHandle handle_forward;
  	cufftReal *Input_fft;
  	cufftComplex *Output_fft;
	cudaMalloc((void**)&Input_fft,  DATASIZE * batch * sizeof(cufftReal) );
	cudaMalloc((void**)&Output_fft, size_fft * batch * sizeof(cufftComplex) );

	cudaMemcpy(Input_fft, data, DATASIZE * batch * sizeof(float), cudaMemcpyHostToDevice);
	cufftPlan1d(&handle_forward, DATASIZE, CUFFT_R2C, batch);
	cufftExecR2C(handle_forward, Input_fft, Output_fft);
	
//---------------------------------------------- Correlation --------------------------------------------------
	cufftComplex *Corr;
	cudaMalloc((void**)&Corr,  size_fft * batch * sizeof(cufftComplex));
	VectorMult<<<DimGrid,DimBlock>>>(Output_fft, Corr, size_fft*batch, size_fft ); 																	    // vector element x element	

	cufftComplex *Input_i;
	cudaMalloc((void**)&Input_i,  size_fft * batch * sizeof(cufftComplex));
	Correlate<<<DimGrid,DimBlock>>>(Corr, Input_i, size_fft*batch ); 				//Todos contra todos

//------------------------------------------------cuda_fft_i---------------------------------------------------
	cufftHandle handle_inverse;
	cufftReal *Output_i;
	cudaMalloc((void**)&Output_i,  DATASIZE * batch * sizeof(cufftReal) );

	cufftPlan1d( &handle_inverse, DATASIZE, CUFFT_C2R, batch);
	cufftExecC2R(handle_inverse, Corr, Output_i);

	cufftReal *XCorr = (cufftReal*)malloc((DATASIZE) * batch * sizeof(cufftReal)); 
	cudaMemcpy(XCorr, Output_i, DATASIZE * batch * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    for (int i=0; i<batch; i++)
		printf(" hostOutputPowerT[%d] = %f\n",i, XCorr[DATASIZE*i]/2);

	cudaEventRecord( stop, 0) ;
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop);
	printf("\n Time: %f ms\n",elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cufftDestroy(handle_inverse);
	cudaFree(Output_i);
	cudaFree(Corr);

	cufftDestroy(handle_forward);
	cudaFree(Input_fft);
	cudaFree(Output_fft);

	free(data);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return (EXIT_SUCCESS);
}


// hostOutputPowerT[0] = 771441344.000000
// hostOutputPowerT[1] = 192102768.000000


/*
	cufftComplex *print = (cufftComplex*)malloc( MAX*count*sizeof(cufftComplex));
	cudaMemcpy( print, Corr, MAX*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost );
	FILE *file;
	char filename[] = "Graph.dat";
	file = fopen(filename, "w");
	int l;
	for( l = 0; l<MAX*count; l++ )
		fprintf(file, "%f    %f\n", print[l].x, print[l].y);
*/
