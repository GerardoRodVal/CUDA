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

#define GRID_DIMENSION  3
#define BLOCK_DIMENSION 3
#define COMP (batch*(batch-1))/2

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

float Maxim( cufftComplex *vector, int size )
{

	float *max;
	max = (float*)malloc(sizeof(float));

	max[0] = vector[0].x;

	for( int i=1; i<size; i++ ){
		if( max[0] < vector[i].x )
			max[0] = vector[i].x;
	}

	float Output = max[0];
	//Output = (float*)malloc(sizeof(float));

	//memcpy( Output, &max[0], sizeof(cufftComplex));

	return Output;

}



cufftComplex *Correlate( cufftComplex *s1, cufftComplex *s2, int xlen, int npts, float dsor )
{

	cufftComplex *Input_i;
	Input_i = (cufftComplex*)malloc( xlen*sizeof(cufftComplex));

	int i;
	for( i=0; i<xlen; i++ ){
		Input_i[i].x = s1[i].x * s2[i].x ;
		Input_i[i].y = s1[i].y * s2[i].y ;
	}

	cufftComplex *Output_i;
	Output_i = (cufftComplex*)malloc( xlen*sizeof(cufftComplex));

	cufftHandle plan_i;							// settings plan to fft
	cufftComplex *data_fft_i;
	cufftReal *dev_dat_i;

	int rank_i = -1;                            
  	int n[] = { xlen };                      
  	int istride = 1, ostride = 1;            
  	int idist = xlen, odist = ( xlen / 2 + 1); 
  	int inembed[] = { 0 };                   
  	int onembed[] = { 0 };                   
  	int batch = npts;   

	cudaMalloc((void**)&dev_dat_i,   xlen*sizeof(cufftReal) );
	cudaMalloc((void**)&data_fft_i,  xlen*sizeof(cufftComplex));

	cudaMemcpy(dev_dat_i, Input_i,   xlen*sizeof(cufftComplex), cudaMemcpyHostToDevice);									
	cufftPlanMany(&plan_i, rank_i, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan_i, dev_dat_i, data_fft_i);

	gpuErrchk(cudaMemcpy(Output_i, data_fft_i,  xlen*sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	cufftComplex *Output;
	Output = (cufftComplex*)malloc( xlen*sizeof(cufftComplex));


	for( int div=0; div<xlen; div++ ){
		Output[div].x = Output_i[div].x/dsor;
		Output[div].y = Output_i[div].y/dsor;
	}

	cudaFree(dev_dat_i);
	cudaFree(data_fft_i);
	cufftDestroy(plan_i);

	return Output;

}


cufftComplex *Vector( cufftComplex *vectorIn, int inicio, int final, int size)
{
	cufftComplex *Output;
	Output = (cufftComplex*)malloc(sizeof(cufftComplex)*size);

	int i;
	int ind = 0;
	for( i=inicio; i<final; i++ ){
		Output[ind] = vectorIn[i];
		ind += 1;
	}

	return Output;
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

	int n1 = nlen;
    int n2 = nlen;

    if( n1 > n2 ){
    	printf( "Reference signal S1 cannot be longer than target S2\n" );
        exit(0);
    }

    int nx = n2-n1+1;
	int nfft;
    nfft = NextPower2( n2+n1 );

// --------------------------------------cuda_fft---------------------------------------------------
	cufftHandle plan;							// settings plan to fft
	cufftComplex *data_fft;
	cufftComplex *Out_fft;
	cufftReal *dev_dat;

	int rank = 1;                            
  	int n[] = { nlen };                      
  	int istride = 1, ostride = 1;            
  	int idist = MAX, odist = ( nlen / 2 + 1); 
  	int inembed[] = { 0 };                   
  	int onembed[] = { 0 };                   
  	int size_fft = (nlen );
  	int batch = count;   

	gpuErrchk(cudaMalloc((void**)&dev_dat, MAX*count*sizeof(cufftReal) ));
	gpuErrchk(cudaMalloc((void**)&data_fft, size_fft*count*sizeof(cufftComplex) ));
	Out_fft = (cufftComplex*)malloc( size_fft * count * sizeof(cufftComplex));
	gpuErrchk(cudaMemcpy(dev_dat, data, MAX*count*sizeof(float), cudaMemcpyHostToDevice));
								
	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan, dev_dat, data_fft);

	cudaMemcpy(Out_fft, data_fft, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	
//------------------------------------Complex conjugate--------------------------------------------------------
	int grid_size  = GRID_DIMENSION;
    int block_size = BLOCK_DIMENSION;

    dim3 DimGrid(grid_size, grid_size, grid_size);
    dim3 DimBlock(block_size, block_size, block_size);

    cufftHandle handle;

    cufftReal *ComCon_d;
	cufftComplex *ComCon_dO;
	cufftComplex *Out_conj; 
	cufftComplex *fft_conj;

	Out_conj = (cufftComplex*)malloc(  nlen * count * sizeof(cufftComplex));
	cudaMalloc((void**)&ComCon_d,      nlen * count * sizeof(cufftReal));
	cudaMalloc((void**)&ComCon_dO,     nlen * count * sizeof(cufftComplex));
    cudaMalloc((void**)&fft_conj,      nlen * count * sizeof(cufftComplex));

	cudaMemcpy(ComCon_d, data_fft,     nlen*count*sizeof(cufftReal), cudaMemcpyDeviceToDevice);

    cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

	cufftExecR2C(handle, ComCon_d, ComCon_dO);
	cudaMemcpy(fft_conj, ComCon_dO, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

	ComplexConj<<<DimGrid,DimBlock>>>( (nlen)*count, fft_conj );

	cudaMemcpy(Out_conj, fft_conj, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

//-------------------------------------------- Correlation --------------------------------------------------------

	cufftComplex *fft_A;
	cufftComplex *fft_B;

	fft_A = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));
	fft_B = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));

	cufftComplex *Con_A;
	cufftComplex *Con_B;

	Con_A = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));
	Con_B = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));

	cufftComplex *Corr_A;
	cufftComplex *Corr_B;

	Corr_A = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));
	Corr_B = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));

	float Power_A;
	float Power_B;

	cufftComplex *Correlation;	
	Correlation = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));

	float Corr_Max;	
	float res;

	float res1 = (float) batch;
	int begin = 0;
	int end = MAX;
	int foot = MAX;
	int n3 = 1;

	FILE *file;
	char filename[] = "Correlations.dat";
	file = fopen(filename, "w");

	for( int x=0; x<batch-1; x++ ){

		fft_A = Vector( Out_fft, begin, end, MAX );
		Con_A = Vector( Out_conj, begin, end, MAX );

		Corr_A = Correlate(fft_A, Con_A, nlen, batch, res1); 
		Power_A = Maxim( Corr_A, MAX );

		int begin2 =  0 + foot;
		int end2 = MAX + foot;

		for( int y=0; y<batch-n3; y++ ){

			fft_B = Vector( Out_fft, begin2, end2, MAX );
			Con_B = Vector( Out_fft, begin2, end2, MAX );

			Corr_B = Correlate( fft_B, Con_B, nlen, batch, res1 ); 
			Power_B = Maxim( Corr_B, MAX );

			res = batch*sqrt(Power_A*Power_B);
			Correlation = Correlate( fft_A, Con_B, MAX, batch, res ); 

			Corr_Max = Maxim( Correlation, MAX );

			//if( Corr_Max > 0.6 )
			//	fprintf(file, "Waveform1 =     Waveform2 =      CC = %f \n", Corr_Max);
			printf("Waveform1 =     Waveform2 =      CC = %f \n", Corr_Max);

			begin2 += MAX;
			end2 += MAX;
		}

		n3 += 1;
		begin += MAX;
		end += MAX;
		foot += MAX;
	}

	cufftDestroy(handle);
	cudaFree(Out_fft);
	cudaFree(Out_conj);
	cudaFree(ComCon_dO);
	cudaFree(ComCon_d);
	cudaFree(fft_conj);
	cudaFree(dev_dat);
	cudaFree(data_fft);
	cufftDestroy(plan);
	free(data);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return (EXIT_SUCCESS);

}