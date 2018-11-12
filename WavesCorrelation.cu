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
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if (uniqueid < nelem){
		array[uniqueid].y = array[uniqueid].y*-1;
 	 }
}


__global__ void Vector( cufftComplex *vectorIn, cufftComplex *Output, int inicio, int final, int size)
{
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if( uniqueid >= inicio and uniqueid < final ){
		Output[uniqueid].x = vectorIn[uniqueid].x;
		Output[uniqueid].y = vectorIn[uniqueid].y;
	}
}

__global__ void VectorMult( cufftComplex *s1, cufftComplex *s2, cufftComplex *Output, int xlen )
{
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if( uniqueid < xlen ){
		Output[uniqueid].x = s1[uniqueid].x * s2[uniqueid].x ;
		Output[uniqueid].y = s1[uniqueid].y * s2[uniqueid].y ;
	}
}


__global__ void Div( cufftComplex *Input, cufftComplex *Output, int max, float dsor )
{
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if (uniqueid < max){
		Output[uniqueid].x = Input[uniqueid].x/dsor ;
		Output[uniqueid].y = Input[uniqueid].y/dsor ;
 	 }
}



__global__ void Maximo( cufftComplex *Input, cufftComplex *Output, int max, int pasos )
{
	int NumThread = blockDim.x*blockDim.y*blockDim.z;
	int idThread  = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
	int BlockId   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

	int uniqueid  = idThread + NumThread*BlockId;

	if( uniqueid >= pasos*0 and uniqueid < pasos*1 ){
		if( Input[uniqueid].x > Output[0].x ){
			Output[0].x = Input[uniqueid].x;
		}
	}	

	if( uniqueid >= pasos*1 and uniqueid < pasos*2 ){
		if( Input[uniqueid].x > Output[1].x ){
			Output[1].x = Input[uniqueid].x;
		}
	}

	if( uniqueid >= pasos*2 and uniqueid < pasos*3 ){
		if( Input[uniqueid].x > Output[2].x ){
			Output[2].x = Input[uniqueid].x;
		}
	}

	if( uniqueid >= pasos*3 and uniqueid < pasos*4 ){
		if( Input[uniqueid].x > Output[3].x ){
			Output[3].x = Input[uniqueid].x;
		}
	}


}



int main(int argc, char **argv) 
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
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

	cudaMalloc((void**)&dev_dat,   size_fft*count*sizeof(cufftReal) );
	cudaMalloc((void**)&data_fft,  size_fft*count*sizeof(cufftComplex) );
	cudaMalloc((void**)&Out_fft,  size_fft*count*sizeof(cufftComplex) );

	gpuErrchk(cudaMemcpy(dev_dat, data, MAX*count*sizeof(float), cudaMemcpyHostToDevice));
								
	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan, dev_dat, data_fft);

	cudaMemcpy(Out_fft, data_fft, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
	
//------------------------------------Complex conjugate--------------------------------------------------------
	int grid_size  = GRID_DIMENSION;
    int block_size = BLOCK_DIMENSION;

    dim3 DimGrid(grid_size, grid_size, grid_size);
    dim3 DimBlock(block_size, block_size, block_size);

    cufftHandle handle;

    cufftReal *ComCon_d;
	cufftComplex *ComCon_dO;
	cufftComplex *fft_conj;
	cufftComplex *Out_conj; 

	cudaMalloc((void**)&ComCon_d,  nlen * count * sizeof(cufftReal));
	cudaMalloc((void**)&ComCon_dO, nlen * count * sizeof(cufftComplex));
    cudaMalloc((void**)&fft_conj,  nlen * count * sizeof(cufftComplex));
    cudaMalloc((void**)&Out_conj,  nlen * count * sizeof(cufftComplex));

	cudaMemcpy(ComCon_d, data, nlen*count*sizeof(cufftReal), cudaMemcpyHostToDevice);

    cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(handle, ComCon_d, ComCon_dO);
	cudaMemcpy(fft_conj, ComCon_dO, nlen*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

	cudaEventRecord(start, 0);
	ComplexConj<<< DimGrid,DimBlock >>>( nlen*count, fft_conj );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	cudaMemcpy(Out_conj, fft_conj, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

//-------------------------------------------- Correlation --------------------------------------------------------
//----------------------------  Variables  ------------------------------
	cufftComplex *fft_A;
	cufftComplex *fft_B;

	cudaMalloc((void**)&fft_A, MAX * sizeof(cufftComplex));
	cudaMalloc((void**)&fft_B, MAX * sizeof(cufftComplex));
	
	cufftComplex *Con_A;
	cufftComplex *Con_B;

	cudaMalloc((void**)&Con_A, MAX * sizeof(cufftComplex));
	cudaMalloc((void**)&Con_B, MAX * sizeof(cufftComplex));

	cufftComplex *Corr_A;
	cufftComplex *Corr_B;
	cufftComplex *Corr_AB;

	cudaMalloc((void**)&Corr_A,  MAX * sizeof(cufftComplex));
	cudaMalloc((void**)&Corr_B,  MAX * sizeof(cufftComplex));
	cudaMalloc((void**)&Corr_AB, MAX * sizeof(cufftComplex));

	float *Power_A;
	float *Power_B;
	float *CC;

	cudaMalloc((void**)&Power_A, sizeof(float));
	cudaMalloc((void**)&Power_B, sizeof(float));
	cudaMalloc((void**)&CC, sizeof(float));

// ----------------------- div variables -----------------------------

	cufftComplex *Input_max;
	cudaMalloc((void**)&Input_max,  MAX * sizeof(cufftComplex) );

// ----------------------- General variables -------------------------

	int begin = 0;
	int end = MAX;
	int foot = MAX;
	int n3 = 1;

	FILE *file;
	char filename[] = "Correlations.dat";
	file = fopen(filename, "w");

// --------------------- ifft variables -------------------------------
	cufftHandle plan_i;						
	cufftComplex *data_fft_i;
	cufftReal *dev_dat_i;

	int rank_i = -1; 
	int n_i[] = { MAX };                      
	int istride_i = 1, ostride_i = 1;   
	int idist_i = MAX, odist_i = ( MAX / 2 + 1); 
	int inembed_i[] = { 0 };                   
	int onembed_i[] = { 0 };                   
	int batch_i = 1;

	cudaMalloc((void**)&dev_dat_i,  MAX * sizeof(cufftReal) );
	cudaMalloc((void**)&data_fft_i, MAX * sizeof(cufftComplex));

	cufftComplex *Output_i;
	cudaMalloc((void**)&Output_i,   MAX * sizeof(cufftComplex) );

// ------------------------------------------------------------------
	cufftComplex *Input_max_H;
	Input_max_H = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));

	for( int A=0; A<count-1; A++ ){

		Vector<<<DimGrid,DimBlock>>>( Out_fft,  fft_A, begin, end, MAX );																    // define vector
		Vector<<<DimGrid,DimBlock>>>( Out_conj, Con_A, begin, end, MAX );
		VectorMult<<<DimGrid,DimBlock>>>(fft_A, Con_A, Corr_A, nlen ); 																	    // vector element x element

		cudaMemcpy(dev_dat_i, Corr_A,  MAX  * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);												// ifft
		cufftPlanMany(&plan_i, rank_i, n_i, inembed_i, istride_i, idist_i, onembed_i, ostride_i, odist_i, CUFFT_R2C, batch_i);
		cufftExecR2C(plan_i, dev_dat_i, data_fft_i);
		cudaMemcpy(Output_i, data_fft_i, MAX * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

		Div<<<DimGrid,DimBlock>>>( Output_i, Input_max, MAX, (float)MAX );																			// div element




		cufftComplex *SalidaMAX;
		cudaMalloc((void**)&SalidaMAX, 4*sizeof(cufftComplex) );
		Maximo<<<DimGrid,DimBlock>>>( Corr_A, SalidaMAX, MAX, 256 );
		cufftComplex *Salida = (cufftComplex*)malloc( 4*sizeof(cufftComplex));
		cudaMemcpy( Salida, SalidaMAX, 4*sizeof(cufftComplex), cudaMemcpyDeviceToHost );
		for( int it=0; it<4	; it++ )
			printf("%f\n", Salida[it].x);




		cudaMemcpy( Input_max_H, Corr_A, MAX*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		float maxA = 0;
		int index =  0;
		for( int i=0; i<MAX; i++ ){
			if( Input_max_H[i].x > maxA ){
				maxA = Input_max_H[i].x;
				index = i;
			}
		}



		printf("Power A: %f   index: %i\n", maxA, index);

		//cudaMemcpy( Power_A, maxA, sizeof(float), cudaMemcpyHostToDevice);

/*
		int begin2 =  0 + foot;
		int end2 = MAX + foot;

		for( int y=0; y<batch-n3; y++ ){

			Vector<<<DimGrid,DimBlock>>>( Out_fft,  fft_B, begin2, end2, MAX );																    // define vector
			Vector<<<DimGrid,DimBlock>>>( Out_conj, Con_B, begin2, end2, MAX );

			VectorMult<<<DimGrid,DimBlock>>>(fft_B, Con_B, Corr_B, nlen ); 																	    // vector element x element

			cudaMemcpy(dev_dat_i, Corr_B,  MAX  * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);												// ifft
			cufftPlanMany(&plan_i, rank_i, n_i, inembed_i, istride_i, idist_i, onembed_i, ostride_i, odist_i, CUFFT_R2C, batch_i);
			cufftExecR2C(plan_i, dev_dat_i, data_fft_i);
			cudaMemcpy(Output_i, data_fft_i, MAX * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

			Div<<<DimGrid,DimBlock>>>( Output_i, Input_max, MAX, MAX );																			// div element

			cudaMemcpy( Input_max_H, Input_max, MAX*sizeof(cufftComplex), cudaMemcpyDeviceToHost ); 

			float maxB;
			maxB = Input_max_H[0].x;

			for( int i=1; i<MAX; i++ ){
				if( maxB < Input_max_H[i].x )
					maxB = Input_max_H[i].x;
			}

			cudaMemcpy( Power_B, &maxB, sizeof(float), cudaMemcpyHostToDevice);

// ----------------------------------------------- Correlation AB --------------------------------------------------------------

			VectorMult<<<DimGrid,DimBlock>>>(fft_A, Con_B, Corr_AB, nlen ); 																	    // vector element x element

			cudaMemcpy(dev_dat_i, Corr_AB,  MAX  * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);												// ifft
			cufftPlanMany(&plan_i, rank_i, n_i, inembed_i, istride_i, idist_i, onembed_i, ostride_i, odist_i, CUFFT_R2C, batch_i);
			cufftExecR2C(plan_i, dev_dat_i, data_fft_i);
			cudaMemcpy(Output_i, data_fft_i, MAX * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

			//float npts = MAX*sqrt(Power_A*Power_B);
			Div<<<DimGrid,DimBlock>>>( Output_i, Input_max, MAX, MAX );																	   // div element

			cudaMemcpy( Input_max_H, Input_max, MAX*sizeof(cufftComplex), cudaMemcpyDeviceToHost ); 

			float maxAB;
			maxAB = Input_max_H[0].x;

			for( int i=1; i<MAX; i++ ){
				if( maxAB < Input_max_H[i].x )
					maxAB = Input_max_H[i].x;
			}

			cudaMemcpy( CC, &maxAB, sizeof(float), cudaMemcpyHostToDevice);

			//if( CC > 0.6 )
			//	fprintf(file, "Waveform1 =     Waveform2 =      CC = %f \n", CC);
			printf("%.4f %.4f  %.4f\n", maxA, maxB, maxAB);

			begin2 += MAX;
			end2 += MAX;
		}
*/
		n3 += 1;
		begin += MAX;
		end += MAX;
		foot += MAX;
	}

	cufftDestroy(handle);
	cudaFree(fft_conj);
	cudaFree(Out_conj);
	cudaFree(ComCon_dO);
	cudaFree(ComCon_d);

	cufftDestroy(plan);
	cudaFree(Out_fft);
	cudaFree(dev_dat);
	cudaFree(data_fft);

	cufftDestroy(plan_i);
	cudaFree(Output_i);
	cudaFree(dev_dat_i);
	cudaFree(data_fft_i);

	free(data);
	
	cudaDeviceSynchronize();
	cudaDeviceReset();
	
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);


	return (EXIT_SUCCESS);

}
