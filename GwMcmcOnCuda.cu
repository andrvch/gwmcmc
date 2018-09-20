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

/* Functions and Kernels: */
__host__ __device__ int PriorCondition ( const Walker wlkr )
{
  int cndtn = 1;
  cndtn = cndtn * ( 5.5 < wlkr.par[TINDX] ) * ( wlkr.par[TINDX] < 6.5 );
  cndtn = cndtn * ( log10 ( 8. / 13. / 6000. ) < wlkr.par[RINDX1] ) * ( wlkr.par[RINDX1] < log10f ( 20. / 13. / 100. ) );
  cndtn = cndtn * ( log10 ( 100. ) < wlkr.par[DINDX1] ) * ( wlkr.par[DINDX1] < log10f ( 6000. ) );
  cndtn = cndtn * ( 0. < wlkr.par[NHINDX] );
  return cndtn;
}

__host__ __device__ float PriorStatistic ( const Walker wlkr, const int cndtn, const float nhMd, const float nhSg )
{
  float prr = 0, sum = 0;
  //float theta = powf ( nhSg, 2 ) / nhMd;
  //float kk = nhMd / theta;
  //sum = sum + ( kk - 1 ) * logf ( wlkr.par[NHINDX] ) - wlkr.par[NHINDX] / theta;
  sum = sum + powf ( ( wlkr.par[NHINDX] - nhMd ) / nhSg, 2 );
  if ( cndtn ) { prr = sum; } else { prr = INF; }
  return prr;
}

__global__ void AssembleArrayOfPriors ( const int nmbrOfWlkrs, const Walker *wlkrs, const float *nhMd, const float *nhSg, float *prrs )
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nmbrOfWlkrs )
  {
    prrs[i] = PriorStatistic ( wlkrs[i], PriorCondition ( wlkrs[i] ), nhMd[i], nhSg[i] );
  }
}

__host__ int Priors ( const Model *mdl, const int nmbrOfWlkrs, const Walker *wlkrs, float *nhMd, float *nhSg, float *prrs )
{
  //LinearInterpolation <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, mdl[0].nmbrOfDistBins, DINDX1, mdl[0].Dist, mdl[0].EBV, mdl[0].errEBV, wlkrs, nhMd, nhSg );
  LinearInterpolationNoErrors <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist1, mdl[0].EBV1, wlkrs, nhMd, nhSg );
  AssembleArrayOfPriors <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, wlkrs, nhMd, nhSg, prrs );
  return 0;
}

__host__ __device__ float GregoryLoredo ( const float tms, const Walker wlkr, const float Ttot, const int N )
{
    float sttstc = 0, f, phi, jt, jtFr, jtInt, jtJt, A;
    f = wlkr.par[0] * 1.E-6 + F0;
    phi = wlkr.par[1];
    jt = 1 + ( NTBINS / ( 2 * PI ) ) * fmodf ( 2 * PI * f * tms + phi, 2 * PI );
    jtFr = modff( jt, &jtInt );
    jtJt = jt - jtFr;
    int jIndx = llroundf ( jtJt );
    A = SumOfComponents ( wlkr ) / NTBINS;
    sttstc = logf ( NTBINS * A ) - A * Ttot / N + logf ( wlkr.par[jIndx] / NTBINS / A );
    return sttstc;
}

__global__ void AssembleArrayOfTimesStatistic ( const int nmbrOfWlkrs, const int nmbrOfPhtns, const float srcExptm, const Walker *wlk, const float *arrTms, float *tmsSttstcs )
{
  int a = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t = a + w * nmbrOfPhtns;
  if ( ( a < nmbrOfPhtns ) && ( w < nmbrOfWlkrs ) )
  {
    tmsSttstcs[t] = GregoryLoredo ( arrTms[a], wlk[w], srcExptm, nmbrOfPhtns );
  }
}

/**
 * Host main routine
 */
int main ( int argc, char *argv[] )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  const int verbose = 1;
  const float lwrNtcdEnrg1 = 0.4;
  const float hghrNtcdEnrg1 = 7.0;
  const float dlt = 1.E-4;
  const float phbsPwrlwInt[NPRS] = { 6.0, log10f ( 1. / 1000. ), 3., 1.5, -5., 1.5, -5., 0.9, -5., 0.9, -5., 0.9, -5., 0.2 };

  /* Initialize */
  Cuparam cdp[NSPCTR];
  Model mdl[NSPCTR];
  Chain chn[NSPCTR];
  Spectrum spc[NSPCTR];

  cdp[0].dev = atoi( argv[1] );
  const char *spcFl1 = argv[2];
  const char *spcFl2 = argv[3];
  const char *spcLst[NSPCTR] = { spcFl1, spcFl2 };

  chn[0].thrdNm = argv[NSPCTR+2];
  chn[0].nmbrOfWlkrs = atoi ( argv[NSPCTR+3] );
  chn[0].nmbrOfStps = atoi ( argv[NSPCTR+4] );
  chn[0].thrdIndx = atoi ( argv[NSPCTR+5] );
  chn[0].dlt = dlt;

  for ( int i = 0; i < NSPCTR; i++ )
  {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg1;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg1;
  }

  InitializeCuda ( cdp );
  InitializeModel ( mdl );
  InitializeChain ( cdp, phbsPwrlwInt, chn );

  SpecInfo ( spcLst, verbose, spc );
  SpecAlloc ( chn, spc );
  SpecData ( cdp, verbose, mdl, spc );

  /* Initialize walkers */
  if ( chn[0].thrdIndx == 0 )
  {
    InitAtRandom ( cdp, chn );
    Priors ( mdl, chn[0].nmbrOfWlkrs, chn[0].wlkrs, chn[0].nhMd, chn[0].nhSg, chn[0].prrs );
    for ( int i = 0; i < NSPCTR; i++ )
    {
      ModelFluxes ( mdl, chn[0].nmbrOfWlkrs, chn[0].wlkrs, i, spc[i] );
      FoldModel ( cdp, chn[0].nmbrOfWlkrs, spc[i] );
      Stat ( chn[0].nmbrOfWlkrs, spc[i] );
      SumUpStat ( cdp, 1, chn[0].nmbrOfWlkrs, chn[0].sttstcs, spc[i] );
    }
  }
  else if ( chn[0].thrdIndx > 0 )
  {
    InitFromLast ( chn );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  /* Run chain */
  printf ( ".................................................................\n" );
  printf ( " Start ...                                                  \n" );

  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].rndmVls, chn[0].nmbrOfRndmVls );

  int stpIndx = 0, sbstIndx;
  while ( stpIndx < chn[0].nmbrOfStps )
  {
    sbstIndx = 0;
    while ( sbstIndx < 2 )
    {
      Propose ( stpIndx, sbstIndx, chn );
      Priors ( mdl, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].nhMd, chn[0].nhSg, chn[0].prpsdPrrs );
      for ( int i = 0; i < NSPCTR; i++ )
      {
        ModelFluxes ( mdl, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, i, spc[i] );
        FoldModel ( cdp, chn[0].nmbrOfWlkrs / 2, spc[i] );
        Stat ( chn[0].nmbrOfWlkrs / 2, spc[i] );
        SumUpStat ( cdp, 1, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdSttstcs, spc[i] );
      }
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
  if ( cdp[0].cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error: Direct Plan configuration failed" ); return 1; }
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

  /* Elapsed time */
  printf ( ".................................................................\n" );
  printf ( " Time to generate: %3.1f ms\n", chn[0].elapsedTime );
  printf ( " Time to compute Autocorrelation Function: %3.1f ms\n", chn[0].cufftElapsedTime );
  printf ( "\n" );

  /* Write results to a file */
  SimpleWriteDataFloat ( "Autocor.out", chn[0].nmbrOfStps, chn[0].atCrrFnctn );
  SimpleWriteDataFloat ( "AutocorCM.out", chn[0].nmbrOfStps, chn[0].cmSmAtCrrFnctn );
  WriteChainToFile ( chn[0].thrdNm, chn[0].thrdIndx, chn[0].nmbrOfWlkrs, chn[0].nmbrOfStps, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs, chn[0].chnOfPrrs );

  /* Destroy cuda related contexts and things: */
  DestroyAllTheCudaStaff ( cdp );

  /* Free memory: */
  FreeSpec ( spc );
  FreeChain ( chn );
  FreeModel ( mdl );

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
