#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <cuda.h>
#include <cufft.h>
extern "C" 
{
	#include <sacio.h>
	#include <sac.h>
}
#define MAX 400000
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void GPUmemo( float *data, int pts )
{
	__shared__ float* trace;
	trace = (float *)malloc(pts*sizeof(float));

}


int main(int argc, char **argv) 
{
//----------------------------------settings to sac -----------------------------------------
	//size_t len = 0;
  	int count = 0;
  	//size_t line_size = 100;
  	int nlen, nerr, max = MAX;
  	char kname[31];
  	//char filename[31];
	float *data;
	//char *line;
	float yarray[MAX], beg, del;
	//FILE *fid;

	//line = (char  *)malloc( line_size * sizeof(char) );
	data = (float *)malloc( 20 * MAX * sizeof(float));	

	//fid = fopen(filename,"r");

// reading sac files
	struct dirent *de;  
	DIR *dr = opendir(".");								//open currently directory
    if (dr == NULL)
    {
		printf("Error open directory" );
    	return 0;
    }

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

			//fprintf(stderr,"Reading SUCCESS: %s\n",kname);
			//fprintf(stderr,"Number of samples read: %d\n\n",nlen);

			memcpy(&data[count*MAX], yarray, nlen*sizeof(float));
			count ++;
		}	 
	}

// --------------------------------------cuda_fft---------------------------------------------------
	nlen = 32768;                               //prueba
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

	cudaMalloc((void**)&dev_dat, MAX*count*sizeof(cufftReal) );
	cudaMalloc((void**)&data_fft, size_fft*count*sizeof(cufftComplex) );
	outfft = (cufftComplex*)malloc( size_fft * count * sizeof(cufftComplex));
	cudaMemcpy(dev_dat, data, MAX*count*sizeof(float), cudaMemcpyHostToDevice);

	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	cufftExecR2C(plan, dev_dat, data_fft);

	cudaMemcpy(outfft, data_fft, size_fft*count*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	GPUmemo<<<count,nlen>>>( dev_dat, nlen );

	cudaFree(dev_dat);
	cudaFree(data_fft);
	cufftDestroy(plan);
	free(data);
	cudaDeviceReset();
  	return (0);
}
