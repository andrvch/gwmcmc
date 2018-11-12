#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__host__ void statisticsLoop ( const int n, const float *r, float *p ) {
  for ( int i = 0; i < n; i++ ) {
    p[i] = ( powf ( r[i], 2 ) + powf ( r[i+1], 2 ) < 1 ) * 1.;
  }
}

__host__ float statSum ( const int n, const float *p ) {
  float res = 0;
  for ( int i = 0; i < n; i++ ) {
    res += p[i];
  }
  return res;
}


int main ( int argc, char *argv[] ) {
  const int n = atoi ( argv[1] );

  curandGenerator_t gen;
  cudaEvent_t start, stop;

  cudaEventCreate ( &start );
  cudaEventCreate ( &stop );
  cudaEventRecord ( start, 0 );

  curandCreateGeneratorHost ( &gen, CURAND_RNG_PSEUDO_DEFAULT );
  curandSetPseudoRandomGeneratorSeed ( gen, 1234ULL );

  float *r, *p, res;
  cudaMallocManaged ( ( void ** ) &r, 2 * n * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &p, n * sizeof ( float ) );

  curandGenerateUniform ( gen, r, 2 * n );
  statisticsLoop ( n, r, p );
  res = statSum ( n, p );

  float elapsedTime;
  cudaEventRecord ( stop, 0 );
  cudaEventSynchronize ( stop );
  cudaEventElapsedTime ( &elapsedTime, start, stop );

  printf ( "%1.8f\n", 4 * res / n );
  printf ( " Time to generate: %3.1f ms\n", elapsedTime );


  curandDestroyGenerator ( gen );

  cudaFree ( r );
  cudaFree ( p );

  return 0;
}
