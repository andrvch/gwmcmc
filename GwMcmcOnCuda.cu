#ifndef _GWMCMCCUDA_CU_
#define _GWMCMCCUDA_CU_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <fitsio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cufft.h>
#include "StrctrsAndFnctns.cuh"

__host__ __device__ int PriorCondition ( const Walker w ) {
  int cnd = 1;
  //for ( int i = 0; i <  NPRS; i++ ) {
  //  cnd *=  0. < w.par[i];
  //}
  return cnd;
}

__host__ __device__ float PriorStatistic ( const Walker w, const int cnd ) {
  float p = 0, sum = 0;
  if ( cnd ) { p = sum; } else { p = INF; }
  return p;
}

__global__ void AssembleArrayOfPriors ( const int n, const Walker *w, float *p ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    p[i] = PriorStatistic ( w[i], PriorCondition ( w[i] ) );
  }
}

__host__ int Priors ( const int n, const Walker *wlk, float *prr ) {
  AssembleArrayOfPriors <<< Blocks ( n ), THRDSPERBLCK >>> ( n, wlk, prr );
  return 0;
}

__global__ void AssembleArrayOfStatistic ( const int n, const Walker *wlk, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    stt[i] = pow ( wlk[i].par[0] - wlk[i].par[1], 2. ) / 0.1 + pow ( wlk[i].par[0] + wlk[i].par[1], 2. );
  }
}

__host__ int Statistics ( const int n, const Walker *wlk, float *stt ) {
  AssembleArrayOfStatistic <<< Blocks ( n ), THRDSPERBLCK >>> ( n, wlk, stt );
  return 0;
}


/**
 * Host main routine
 */
int main ( int argc, char *argv[] ) {
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  const int verbose = 1;
  const float dlt = 1.E-6;
  const float p0[NPRS] = { 0.1, 0.1 };

  Cuparam cdp[1];
  Chain chn[1];

  cdp[0].dev = atoi( argv[1] );

  chn[0].thrdNm = argv[2];
  chn[0].nmbrOfWlkrs = atoi ( argv[3] );
  chn[0].nmbrOfStps = atoi ( argv[4] );
  chn[0].thrdIndx = atoi ( argv[5] );
  chn[0].dlt = dlt;

  InitializeCuda ( verbose, cdp );
  InitializeChain ( verbose, cdp, p0, chn );

  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].rndmVls, NPRS * chn[0].nmbrOfWlkrs );

  if ( chn[0].thrdIndx == 0 ) {
    InitAtRandom ( chn );
    Priors ( chn[0].nmbrOfWlkrs, chn[0].wlkrs, chn[0].prrs );
    Statistics ( chn[0].nmbrOfWlkrs, chn[0].wlkrs, chn[0].sttstcs );
  }
  else {
    InitFromLast ( chn );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  printf ( ".................................................................\n" );
  printf ( " Start ...                                                  \n" );

  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].rndmVls, chn[0].nmbrOfRndmVls );

  int stpIndx = 0, sbstIndx;
  while ( stpIndx < chn[0].nmbrOfStps ) {
    sbstIndx = 0;
    while ( sbstIndx < 2 ) {
      Propose ( stpIndx, sbstIndx, chn );
      Priors ( chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].prpsdPrrs );
      Statistics ( chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs );
      Update ( stpIndx, sbstIndx, chn );
      sbstIndx += 1;
    }
    ToChain ( stpIndx, chn );
    stpIndx += 1;
  }
  printf ( "      ... >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done!\n" );

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].elapsedTime, cdp[0].start, cdp[0].stop );

  cudaEventRecord ( cdp[0].start, 0 );

  /* Autocorrelation function */
  int NN[RANK] = { chn[0].nmbrOfStps };
  cdp[0].cufftRes = cufftPlanMany ( &cdp[0].cufftPlan, RANK, NN, NULL, 1, chn[0].nmbrOfStps, NULL, 1, chn[0].nmbrOfStps, CUFFT_C2C, chn[0].nmbrOfWlkrs );
  ReturnChainFunction <<< Grid ( chn[0].nmbrOfWlkrs, chn[0].nmbrOfStps ), dimBlock >>> ( chn[0].nmbrOfStps, chn[0].nmbrOfWlkrs, 0, chn[0].chnOfWlkrs, chn[0].chnFnctn );
  AutocorrelationFunctionAveraged ( cdp[0].cufftRes, cdp[0].cublasStat, cdp[0].cublasHandle, cdp[0].cufftPlan, chn[0].nmbrOfStps, chn[0].nmbrOfWlkrs, chn[0].chnFnctn, chn[0].atCrrFnctn );

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].cufftElapsedTime, cdp[0].start, cdp[0].stop );

  /* Autocorreation time */
  CumulativeSumOfAutocorrelationFunction ( chn[0].nmbrOfStps, chn[0].atCrrFnctn, chn[0].cmSmAtCrrFnctn );
  int MM = ChooseWindow ( chn[0].nmbrOfStps, 5e0f, chn[0].cmSmAtCrrFnctn );
  chn[0].atcTime = 2 * chn[0].cmSmAtCrrFnctn[MM] - 1e0f;

  printf ( ".................................................................\n" );
  printf ( " Autocorrelation time window -- %i\n", MM );
  printf ( " Autocorrelation time -- %.8E\n", chn[0].atcTime );
  printf ( " Autocorrelation time threshold -- %.8E\n", chn[0].nmbrOfStps / 5e1f );
  printf ( " Effective number of independent samples -- %.8E\n", chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps / chn[0].atcTime );
  printf ( ".................................................................\n" );
  printf ( " Time to generate: %3.1f ms\n", chn[0].elapsedTime );
  printf ( " Time to compute Autocorrelation Function: %3.1f ms\n", chn[0].cufftElapsedTime );
  printf ( "\n" );

  /* Write results to a file */
  SimpleWriteDataFloat ( "Autocor.out", chn[0].nmbrOfStps, chn[0].atCrrFnctn );
  SimpleWriteDataFloat ( "AutocorCM.out", chn[0].nmbrOfStps, chn[0].cmSmAtCrrFnctn );
  WriteChainToFile ( chn[0].thrdNm, chn[0].thrdIndx, chn[0].nmbrOfWlkrs, chn[0].nmbrOfStps, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs, chn[0].chnOfPrrs );

  DestroyAllTheCudaStaff ( cdp );
  FreeChain ( chn );

  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits

  cdp[0].err = cudaDeviceReset ( );
  if ( cdp[0].err != cudaSuccess )
  {
    fprintf ( stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString ( cdp[0].err ) );
    exit ( EXIT_FAILURE );
  }

  return 0;
}

#endif // _GWMCMCCUDA_CU_
