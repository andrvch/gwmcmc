#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

__global__ void insideTheUnitCircle ( const int n, const float *r, float *p ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    p[i] = ( powf ( r[i], 2 ) + powf ( r[i+1], 2 ) =< 1 ) * 1.;
  }
}

int main ( int argc, char *argv[] ) {
  const int n = atoi ( argv[1] );
  const int nThr = atoi ( argv[2] );
  const int incxx = 1;

  int dev = 0;

  cudaError_t err = cudaSuccess;
  cublasHandle_t cublasHandle = 0;
  curandGenerator_t gen;
  cudaEvent_t start, stop;

  cudaEventCreate ( &start );
  cudaEventCreate ( &stop );
  cudaEventRecord ( start, 0 );

  cudaSetDevice ( dev );
  cublasCreate ( &cublasHandle );

  curandCreateGenerator ( &gen, CURAND_RNG_PSEUDO_DEFAULT );
  curandSetPseudoRandomGeneratorSeed ( gen, 1234ULL );

  float *r, *p, res;
  cudaMallocManaged ( ( void ** ) &r, 2 * n * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &p, n * sizeof ( float ) );

  dim3 block ( nThr );
  dim3 grid ( ( n + block.x - 1 ) / block.x );

  curandGenerateUniform ( gen, r, 2 * n );

  dim3 block ( nThr );
  dim3 grid ( ( n + block.x - 1 ) / block.x );

  insideTheUnitCircle <<< grid, block >>> ( n, r, p );

  cublasSasum ( cublasHandle, n, p, incxx, &res );

  float elapsedTime;
  cudaEventRecord ( stop, 0 );
  cudaEventSynchronize ( stop );
  cudaEventElapsedTime ( &elapsedTime, start, stop );

  printf ( "%1.8f\n", 4 * res / n );
  printf ( " Time to generate: %3.1f ms\n", elapsedTime );

  cublasDestroy ( cublasHandle );
  curandDestroyGenerator ( gen );

  cudaFree ( r );
  cudaFree ( p );

  err = cudaDeviceReset ();
  if ( err != cudaSuccess ) {
    fprintf ( stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString ( err ) );
    exit ( EXIT_FAILURE );
  }

  return 0;
}
