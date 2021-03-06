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

#define Nsac            5
#define MAX             1024
#define GRID_DIMENSION  32
#define BLOCK_DIMENSION 8

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
    printf("\n");
    printf("            Device Number: %d\n", i);
    printf("              Device name: %s\n",            prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",            prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",            prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

__global__ void Power( cufftComplex *Input, cufftComplex *Output, int nlen, int npts )
{
    int ThreadPerBlock   = blockDim.x*blockDim.y*blockDim.z;
    int ThreadNumInBlock = (threadIdx.x + threadIdx.y*blockDim.x) + threadIdx.z*(blockDim.x*blockDim.y);
    int BlockNumInGrid   = (blockIdx.x + blockIdx.y*gridDim.x) + blockIdx.z*(gridDim.x*gridDim.y);

    int globalThreadNum  = ThreadNumInBlock + ThreadPerBlock*BlockNumInGrid;

    if( globalThreadNum < nlen ){
        Output[globalThreadNum].x = (Input[globalThreadNum].x * Input[globalThreadNum].x + Input[globalThreadNum].y * Input[globalThreadNum].y)/npts;
        Output[globalThreadNum].y = 0;
    }
}

__global__ void Correlation( cufftComplex *Input, cufftComplex *Output, int batch_id, int size, int begin )
{
    int ThreadPerBlock  = blockDim.x*blockDim.y*blockDim.z;
    int index = threadIdx.x + ThreadPerBlock*blockIdx.x;

    //printf("  %i  %i  %i  %i+%i*%i \n",  index+begin, index, index+batch_id*size, index, batch_id, size );

    Output[ index+begin ].x = Input[ index ].x*Input[ index + batch_id*size ].x  + Input[index].y*Input[ index + batch_id*size ].y;
    Output[ index+begin ].y = Input[ index ].x*Input[ index + batch_id*size ].y  - Input[index].y*Input[ index + batch_id*size ].x;
}   

__global__ void Coherence( cufftComplex *Input, cufftComplex *Output, int batch_id, int size, int begin )
{
    int ThreadPerBlock  = blockDim.x*blockDim.y*blockDim.z;
    int index = threadIdx.x + blockIdx.x*ThreadPerBlock;

    Output[ index+begin ].x =  Input[ index ].x * Input[ index+batch_id*size ].x - Input[ index ].x * Input[ index+batch_id*size ].y;
    Output[ index+begin ].y =  Input[ index ].x * Input[ index+batch_id*size ].y - Input[ index ].y * Input[ index+batch_id*size ].x;
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
    char kname[43];
    float *data;
    float yarray[MAX];
    float beg, del;
    char Names[Nsac][43];
    
    data = (float *)malloc( Nsac*MAX*sizeof(float));

    struct dirent *de;  
    DIR *dr = opendir(".");                                                                             //open currently directory
    while ((de = readdir(dr)) != NULL)
    {
        if( strstr( de->d_name, ".sac" ) )                                                              // only sac files
        {
            strcpy( kname , de->d_name );                                                               // reading sac files
            rsac1( kname, yarray, &nlen, &beg, &del, &max, &nerr, strlen( kname ) ) ;           

            if ( nerr != 0 ) 
            {   
                fprintf(stderr, "Error reading SAC file: %s\n", kname);
                exit ( nerr ) ;
            }

            printf("%s file number %i  \n", kname, count );
            strcpy( Names[count], kname );
            memcpy(&data[count*MAX], yarray, nlen*sizeof(float));
            count ++;
        }
    }

    printf("\n");
// ---------------------------------------------fft_settings---------------------------------------------------

    int DATASIZE = MAX;
    int size_fft = DATASIZE / 2 +1;
    int batch = count;    
    cufftHandle handle_forward;
    cufftReal    *Input_fft;
    cufftComplex *Output_fft;

// -----------------------------------------------cuda_fft-----------------------------------------------------

    cudaMalloc((void**)&Input_fft,  DATASIZE * batch * sizeof(cufftReal) );
    cudaMalloc((void**)&Output_fft, DATASIZE * batch * sizeof(cufftComplex) );

    cudaMemcpy(Input_fft, data, DATASIZE * batch * sizeof(cufftReal), cudaMemcpyHostToDevice);
    cufftPlan1d(&handle_forward, DATASIZE, CUFFT_R2C, batch);
    cufftExecR2C(handle_forward, Input_fft, Output_fft);
    
// ----------------------------------------------- Power ------------------------------------------------------

    cufftComplex *Power_Out;
    cudaMalloc((void**)&Power_Out,  size_fft * batch * sizeof(cufftComplex));
    Power<<< DimGrid, DimBlock >>>(Output_fft, Power_Out, size_fft*batch, size_fft);

// --------------------------------------- Correlation and Coherence ------------------------------------------

    int BATCH = (batch*(batch-1))/2;
    int Begin = 0;

    cufftComplex *Correlation_Out;
    cudaMalloc((void**)&Correlation_Out,  size_fft * BATCH * sizeof(cufftComplex) );

    cufftComplex *Coherence_Out;
    cudaMalloc((void**)&Coherence_Out,    size_fft * BATCH * sizeof(cufftComplex) );

    for( int floor=1; floor<batch; floor++ ){
        Correlation<<< batch-floor, size_fft >>>(Output_fft, Correlation_Out, floor, size_fft, Begin);
        Coherence  <<< batch-floor, size_fft >>>(Output_fft, Coherence_Out,   floor, size_fft, Begin);
        Begin += size_fft*(batch-floor);
    }

// ----------------------------------------------- cuda_fft_i ---------------------------------------------------

    cufftHandle handle_inverse;
    cufftHandle handle_inverse2;
   
    cufftReal *Power_Out_i;
    cufftReal *Correlation_out_i;
    cufftReal *Coherence_out_i;

    cudaMalloc((void**)&Power_Out_i,       DATASIZE * batch * sizeof(cufftReal) );
    cudaMalloc((void**)&Correlation_out_i, size_fft * BATCH * sizeof(cufftReal) );
    cudaMalloc((void**)&Coherence_out_i,   size_fft * BATCH * sizeof(cufftReal) );

    cufftPlan1d( &handle_inverse, DATASIZE, CUFFT_C2R, batch);
    cufftPlan1d( &handle_inverse2, size_fft, CUFFT_C2R, batch);

    cufftExecC2R(handle_inverse, Power_Out, Power_Out_i);
    cufftExecC2R(handle_inverse, Correlation_Out, Correlation_out_i);
    cufftExecC2R(handle_inverse, Coherence_Out, Coherence_out_i);

// ------------------------------------------------ Print Results ----------------------------------------------------

    cufftReal *Out_Power = (cufftReal*)malloc( DATASIZE * batch * sizeof(cufftReal));
    cufftReal *Out_Corr  = (cufftReal*)malloc( size_fft * BATCH * sizeof(cufftReal)); 
    cufftReal *Out_Coh   = (cufftReal*)malloc( size_fft * BATCH * sizeof(cufftReal)); 
 
    cudaMemcpy(Out_Power, Power_Out_i,  DATASIZE * batch * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    cudaMemcpy(Out_Corr,  Correlation_out_i, size_fft * BATCH * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    cudaMemcpy(Out_Coh,   Coherence_out_i, size_fft * BATCH * sizeof(cufftReal), cudaMemcpyDeviceToHost);

    float max_corr[BATCH];
    for (int i=0; i < BATCH; i++){
        for (int j =0; j < DATASIZE; j++){
            if (Out_Corr[i*DATASIZE + j] > max_corr[i]){
                max_corr[i] = Out_Corr[i*DATASIZE + j];
            }
        }
    }

    float max_cohr[BATCH];
    for (int i=0; i < BATCH; i++){
        for (int j =0; j < size_fft; j++){
            if (Out_Coh[i*size_fft + j] > max_cohr[i]){
                max_cohr[i] = Out_Coh[i*size_fft + j];
            }
        }
    }
    
    int id_v1 = 0;
    int id_v2 = 1;
    int v_id = id_v2;
    int B = 0;
    int E = batch-1;

    for( int i=0; i<batch; i++ ){
        printf("\n--------------- Power %f of file %d --------------\n", Out_Power[i*DATASIZE]/2000, i );
        for( int j=B; j<E; j++ ){
            //printf("%i  %i \n", id_v1, id_v2);
            printf("\n with file number %d Correlation = %f \n", id_v2, 2*max_corr[j]/(DATASIZE*sqrt(Out_Power[id_v1*DATASIZE]*Out_Power[id_v2*DATASIZE])) );
            printf("                      Coherence = %f \n", max_cohr[j]/(Out_Power[id_v1*DATASIZE]*Out_Power[id_v2*DATASIZE]) );
            id_v1 += 1;
            id_v2 += 1;
        }
        id_v1 = 0;
        id_v2 = v_id+1;
        v_id += 1;
        B += batch-(i+1);
        E += batch-(i+2);
    }

    check_gpu_card_type();

//-------------------------------------------------Finish---------------------------------------------------

    cudaEventRecord( stop, 0) ;
    cudaEventSynchronize(stop);
    float elapsedTime; 
    cudaEventElapsedTime( &elapsedTime, start, stop);
    printf("Time: %f milliseconds\n",elapsedTime/1000);
    printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cufftDestroy(handle_inverse);
    cudaFree(Power_Out_i);
    cudaFree(Correlation_out_i);
    cudaFree(Coherence_out_i);
    
    cufftDestroy(handle_forward);
    cudaFree(Input_fft);
    cudaFree(Output_fft);

    cudaFree(Power_Out);
    cudaFree(Correlation_Out);
    cudaFree(Coherence_Out);

    free(data);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return (EXIT_SUCCESS);
}
