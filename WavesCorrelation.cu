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

#define GRID_DIMENSION  3
#define BLOCK_DIMENSION 3

#define MAX 60001
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


__global__ void Cross_Point( long int nx, cufftComplex *f1, cufftComplex *f2, cufftComplex *InVector )
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

	if (uniqueid < nx){
			InVector[uniqueid].x = f1[uniqueid].x*f2[uniqueid].x;
			InVector[uniqueid].y = f1[uniqueid].y*f2[uniqueid].y;
	 	 }
}




int NextPower2( unsigned int v )
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}


void xcor( cufftComplex *array, int nx )
{

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

	int n1 = 60000;
    int n2 = 60001;

    if( n1 > n2 ){
    	printf( "Reference signal S1 cannot be longer than target S2" );
        exit(0);
    }

    int nx = n2-n1+1;
	int nfft;

    nfft = NextPower2( n2+n1 );

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
  	int size_fft = (nlen );
  	int batch = count;   

	gpuErrchk(cudaMalloc((void**)&dev_dat, MAX*count*sizeof(cufftReal) ));
	gpuErrchk(cudaMalloc((void**)&data_fft, size_fft*count*sizeof(cufftComplex) ));
	outfft = (cufftComplex*)malloc( size_fft * count * sizeof(cufftComplex));
	gpuErrchk(cudaMemcpy(dev_dat, data, MAX*count*sizeof(float), cudaMemcpyHostToDevice));
								
	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan, dev_dat, data_fft);

	gpuErrchk(cudaMemcpy(outfft, data_fft, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	

//------------------------------------Complex conjugate--------------------------------------------------------
	int grid_size  = GRID_DIMENSION;
    int block_size = BLOCK_DIMENSION;

    dim3 DimGrid(grid_size, grid_size, grid_size);
    dim3 DimBlock(block_size, block_size, block_size);

    cufftHandle handle;

    cufftReal *ComCon_d;
	cufftComplex *ComCon_dO;
	cufftComplex *ComCon_hO; 
	cufftComplex *fft_conj;

	ComCon_hO = (cufftComplex*)malloc((nlen) * count * sizeof(cufftComplex));
	cudaMalloc((void**)&ComCon_d, nlen*count*sizeof(cufftReal));
	cudaMalloc((void**)&ComCon_dO, (nlen) * count * sizeof(cufftComplex));
    cudaMalloc((void**)&fft_conj, (nlen) * count * sizeof(cufftComplex));

	cudaMemcpy(ComCon_d, data_fft, nlen*count*sizeof(cufftReal), cudaMemcpyDeviceToDevice);

    cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

	cufftExecR2C(handle, ComCon_d, ComCon_dO);
	cudaMemcpy(fft_conj, ComCon_dO, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

	ComplexConj<<<DimGrid,DimBlock>>>( (nlen)*count, fft_conj );

	cudaMemcpy(ComCon_hO, fft_conj, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

//---------------------------------------- Correlation --------------------------------------------------------

	cufftComplex *OutVector;
	cufftComplex *InVector;

	cudaMalloc((void**)&OutVector, size_fft*batch*sizeof(cufftComplex));
	cudaMalloc((void**)&InVector, size_fft*batch*sizeof(cufftComplex));

	cufftComplex *Out_fft;
	cufftComplex *Out_conj;

	cudaMalloc((void**)&Out_fft, size_fft*batch*sizeof(cufftComplex));
	cudaMalloc((void**)&Out_conj, size_fft*batch*sizeof(cufftComplex));

	cudaMemcpy(Out_conj, ComCon_hO, (nlen)*count*sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(Out_fft, outfft, (nlen)*count*sizeof(cufftComplex), cudaMemcpyHostToDevice);	

	Cross_Point<<<DimGrid,DimBlock>>>( nx , Out_conj, Out_fft, InVector );

	cudaMemcpy(OutVector, InVector, size_fft*batch*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);


	cufftHandle plan_i;							// settings plan to ifft
	cufftComplex *xcor;
	cufftReal *dev_dat_i;

	int rank_i = -1;                            

	gpuErrchk(cudaMalloc((void**)&dev_dat_i, MAX*count*sizeof(cufftReal) ));
	xcor = (cufftComplex*)malloc( size_fft * count * sizeof(cufftComplex));
	gpuErrchk(cudaMemcpy(dev_dat_i, data, MAX*count*sizeof(float), cudaMemcpyHostToDevice));
								
	cufftPlanMany(&plan_i, rank_i, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan_i, dev_dat_i, OutVector);

	gpuErrchk(cudaMemcpy(xcor, OutVector, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	
// ------------------------ scale by sqrt(norm(s1)*norm(s2win)) where s2win is the moving window of s2 --------

	//float s2s2 = 
	int scal[nx] = {0};
	//scal[0] = sum(s2s2[:n1])
	
	/*for(int k=0; k<(nx-1); k++)
        scal[k+1] = scal[k] + s2s2[n1 + k] - s2s2[k]
*/
    // n1 ?

	//xcor(outfft_i, nx);

//-------------------------------------------------------------------------------------------------------------
	cufftDestroy(handle);
	cudaFree(OutVector);
    cudaFree(InVector);
    cudaFree(Out_fft);
	cudaFree(Out_conj);
    cudaFree(ComCon_dO);
    cudaFree(ComCon_d);
    cudaFree(fft_conj);
	cudaFree(dev_dat);
	cudaFree(dev_dat_i);
	cudaFree(data_fft);
	cufftDestroy(plan);
	cufftDestroy(plan_i);
	free(data);
	cudaDeviceSynchronize();
	cudaDeviceReset();
  	return (EXIT_SUCCESS);
}