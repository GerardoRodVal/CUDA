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



float Normalize( float *array, int n1 )
{
	float Vsum;

	for (int i=0; i<n1; i++)
	    Vsum  += array[i]*array[i];
	
	return sqrt(Vsum);
}



void Files_Xcor( cufftComplex *xcor, int batch, int size_fft )
{	
	FILE *file;
	char filename[] = "Correlation_.dat";
	int i;
	for( i = 0; i<(batch*(batch-1))/2; i++ )
	{
		filename[11] = i + '0';
		file = fopen(filename, "w");
		int j;
		for( j = 0; j < size_fft; j++ )
		{
			fprintf(file, "%f    %f\n", xcor[j].x, xcor[j].y );
		}
		fclose(file);
	}
}


void Spect( int N )
{
	FILE   *gnuplot = NULL;
	char Correlation[] = "Correlation_.dat";
	gnuplot=popen("gnuplot","w");
	fprintf(gnuplot,"set term postscript eps enhanced color\n");
	int i;
    for( i=0; i<(N*(N-1))/2; i++ )
    {
                Correlation[11] = i + '0';
                fprintf(gnuplot, "set logscale xz\n");
                fprintf(gnuplot, "set output 'graphics_fft_%i.eps'\n", i);
                fprintf(gnuplot, "plot '%s' u :2 with lines\n", Correlation);
                fprintf(gnuplot, "set output\n");
                fflush(gnuplot);
    }
	pclose(gnuplot);
}


cufftComplex *Vector( cufftComplex *vectorIn, int inicio, int final)
{
	cufftComplex *Output;
	Output = (cufftComplex*)malloc( MAX*sizeof(cufftComplex));

	int ind = 0;
	for( int i=inicio; i<final; i++ ){
		Output[ind] = vectorIn[i];
		ind += 1;
	}	

	return Output;
}


cufftComplex *Comparaciones( cufftComplex *f1, cufftComplex *f2, int N )
{
	cufftComplex *OutputC;

	for( int i=0; i<N; i++ ){
		OutputC[i].x = f1[i].x * f2[i].x;
		OutputC[i].y = f1[i].y * f2[i].y;
	}

	return OutputC;
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
    	printf( "Reference signal S1 cannot be longer than target S2" );
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

	gpuErrchk(cudaMemcpy(Out_fft, data_fft, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	

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

	Out_conj = (cufftComplex*)malloc((nlen) * count * sizeof(cufftComplex));
	cudaMalloc((void**)&ComCon_d, nlen*count*sizeof(cufftReal));
	cudaMalloc((void**)&ComCon_dO, (nlen) * count * sizeof(cufftComplex));
    cudaMalloc((void**)&fft_conj, (nlen) * count * sizeof(cufftComplex));

	cudaMemcpy(ComCon_d, data_fft, nlen*count*sizeof(cufftReal), cudaMemcpyDeviceToDevice);

    cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

	cufftExecR2C(handle, ComCon_d, ComCon_dO);
	cudaMemcpy(fft_conj, ComCon_dO, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

	ComplexConj<<<DimGrid,DimBlock>>>( (nlen)*count, fft_conj );

	cudaMemcpy(Out_conj, fft_conj, (nlen)*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

//-------------------------------------------- Correlation --------------------------------------------------------
	#define COMP (batch*(batch-1))/2

	cufftComplex *xcor;																			// contiene todas las comparaciones 
	xcor = (cufftComplex*)malloc( MAX*COMP*sizeof(cufftComplex));

	int begin = 0;
	int end = MAX;
	int foot = 0;
	int n3 = 1;

	cufftComplex *f1;
	cufftComplex *f2;
	cufftComplex *f1xf2;	

	f1 = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));
	f2 = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));
	f1xf2 = (cufftComplex*)malloc( MAX * sizeof(cufftComplex));


	for( int x=0; x<batch-1; x++ ){
		f1 = Vector( Out_conj, begin, end );
		int begin2 = MAX + foot;
		int end2 = (MAX+MAX) + foot;
		for( int y=0; y<batch-n3; y++ ){
			f2 = Vector( Out_fft, begin2, end2 );
			f1xf2 = Comparaciones( f1, f2, MAX );
			begin2 += MAX;
			end2 += MAX;
			memcpy( xcor, f1xf2, MAX*sizeof(cufftComplex)  );
		}
		n3 += 1;
		begin += MAX;
		end += MAX;
		foot += MAX;
	}
//------------------------------------------------ ifft-----------------------------------------------------------

	cufftComplex *Final_Xcor;
	Final_Xcor = (cufftComplex*)malloc( MAX*COMP*sizeof(cufftComplex));

	cufftComplex *data_fft_i;
	cudaMalloc((void**)&data_fft_i,  MAX*COMP*sizeof(cufftComplex));

	cufftHandle plan_i;							
	cufftReal *dev_dat_i;

	int rank_i = -1;

	gpuErrchk(cudaMalloc((void**)&dev_dat_i, MAX*COMP*sizeof(cufftReal) ));
	gpuErrchk(cudaMemcpy(dev_dat_i, xcor, MAX*count*sizeof(cufftComplex), cudaMemcpyHostToDevice));
									
	cufftPlanMany(&plan_i, rank_i, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan_i, dev_dat_i, data_fft_i);

	gpuErrchk(cudaMemcpy(Final_Xcor, data_fft_i,  MAX*COMP*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	
// ------------------------ scale by sqrt(norm(s1)*norm(s2win)) where s2win is the moving window of s2 -----------

	float s1[n1];
	float s2[n2];
	float s2s2[n2];
	float scal[nx] = {0};											//     scal = np.zeros(nx)

	int size = MAX;
	for( int i=0; i<batch-1; i++ ){

	   	int ind=0;
		for( int j=(size-MAX); j<size; j++ ){				
			s1[ind] = data[j+MAX]; 
			s2[ind] = data[j];
			s2s2[ind] = s2[ind]*s2[ind];
			ind += 1;
		}
		size = size+MAX;

		for( int k=0; k<n1; k++  ){
			scal[0] += s2s2[k];										//    scal[0] = sum(s2s2[:n1])
		}

		for( int l=0; l<nx-1; l++ ){							
			scal[l+1] = scal[l] + s2s2[n1+l]-s2s2[l];				
		}

		float norm;
		norm = Normalize(s1, n1);
			
		for( int m=0; m<nx; m++ ){                              	//     xcor = xcor[:nx]
			scal[m] = sqrt(scal[m]) * norm;
			Final_Xcor[m].x = Final_Xcor[m].x/scal[m];
		}

		Files_Xcor( Final_Xcor, batch, size_fft );
		Spect( batch );
	}

	cufftDestroy(handle);
	cudaFree(Out_fft);
	cudaFree(Out_conj);
	cudaFree(ComCon_dO);
	cudaFree(ComCon_d);
	cudaFree(fft_conj);
	cudaFree(dev_dat);
	cudaFree(dev_dat_i);
	cudaFree(data_fft);
	cudaFree(data_fft_i);
	cufftDestroy(plan);
	cufftDestroy(plan_i);
	free(data);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return (EXIT_SUCCESS);

}