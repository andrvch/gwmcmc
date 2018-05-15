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
#include "GwMcmcStructuresFunctionsAndKernels.cuh"

/* Functions and Kernels: */
__host__ __device__ int PriorCondition ( const Walker wlkr )
{
  int indx, cndtn = 1;
  indx = 0;
  cndtn = cndtn * (   0. < wlkr.par[indx] ) * ( wlkr.par[indx] < 5.5 );
  indx = 1; // pl normalization
  cndtn = cndtn * (  -9. < wlkr.par[indx] );
  indx = 2; // Temperature
  cndtn = cndtn * ( 0.03 < wlkr.par[indx] ) * ( wlkr.par[indx] < 1. );
  indx = 3; // Norm
  cndtn = cndtn * (  -9. < wlkr.par[indx] );
  indx = 4; // Distance
  cndtn = cndtn * (  -1. < wlkr.par[indx] ) * ( wlkr.par[indx] < 3.3 );
  indx = 5;
  cndtn = cndtn * (   0. < wlkr.par[indx] ) * ( wlkr.par[indx] < 5.5 );
  indx = 6; // plnorm
  cndtn = cndtn * (  -9. < wlkr.par[indx] );
  indx = NHINDX; // Hydrogen column density
  cndtn = cndtn * (   0. < wlkr.par[indx] );
  return cndtn;
}

__host__ __device__ float PriorStatistic ( const Walker wlkr, const int cndtn, const float mNh, const float sNh )
{
  float prr = 0, sum = 0, mean = 0, sigma = 0.06;
  float theta = powf ( sNh, 2 ) / mNh;
  float kk = mNh / theta;
  sum = sum + ( kk - 1 ) * logf ( wlkr.par[NHINDX] ) - wlkr.par[NHINDX] / theta;
  //sum = sum + powf ( ( wlkr.par[NHINDX] - mNh ) / sNh, 2 );
  int indx = NHINDX + 1;
  while ( indx < NPRS )
  {
    sum = sum + powf ( ( wlkr.par[indx] - mean ) / sigma, 2 );
    indx += 1;
  }
  if ( cndtn ) { prr = sum; } else { prr = INF; }
  return prr;
}

__global__ void AssembleArrayOfModelFluxes ( const int spIn, const int NW, const int NE, const float *en, const float *arf, const float *absrptn, const Walker *wlk, float *flx )
{
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t = e + w * NE;
  float f;
  if ( ( e < NE ) && ( w < NW ) )
  {
    if ( spIn == 0 )
    {
      f = PowerLaw ( wlk[w].par[0], wlk[w].par[1], en[e], en[e+1] );
      f = f + BlackBody ( wlk[w].par[2], wlk[w].par[3], en[e], en[e+1] );
    }
    else if ( spIn == 1 )
    {
      f = PowerLaw ( wlk[w].par[5], wlk[w].par[6], en[e], en[e+1] );
    }
    flx[t] = f * arf[e] * absrptn[t];
  }
}

__host__ int ModelFluxes ( const Model *mdl, const int nmbrOfWlkrs, const Walker *wlkrs, const int indx, Spectrum spec )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  dim3 dimGrid = Grid ( spec.nmbrOfEnrgChnnls, nmbrOfWlkrs );
  AssembleArrayOfAbsorptionFactors <<< dimGrid, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, ATNMR, spec.crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, wlkrs, spec.absrptnFctrs );
  //BilinearInterpolation <<< dimGrid, dimBlock >>> ( chn[0].nmbrOfWlkrs, spc[i].spec.nmbrOfEnrgChnnls, 2, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spc[0].spec.enrgChnnls, chn[0].wlkrs, spc[0].spec.mdlFlxs );
  AssembleArrayOfModelFluxes <<< dimGrid, dimBlock >>> ( indx, nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, spec.enrgChnnls, spec.arfFctrs, spec.absrptnFctrs, wlkrs, spec.mdlFlxs );
  return 0;
}

__host__ int Priors ( const Model *mdl, Chain *chn )
{
  int blcks = Blocks ( chn[0].nmbrOfWlkrs / 2 );
  LinearInterpolation <<< blcks, THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs / 2, mdl[0].nmbrOfDistBins, 4, mdl[0].Dist, mdl[0].EBV, mdl[0].errEBV, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh );
  AssembleArrayOfPriors <<< blcks, THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh, chn[0].prrs );
  return 0;
}

/**
 * Host main routine
 */
int main ( int argc, char *argv[] )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  const int verbose = 1;
  const float lwrNtcdEnrg = 0.5;
  const float hghrNtcdEnrg = 8.0;
  const float dlt = 1.E-4;
  const float phbsPwrlwInt[NPRS] = { 1.6, log10f ( 7.E-6 ), 0.1, -3., log10f ( 8E2 ), 1.3, log10f ( 7.E-5 ), 0.15 };

  /* Initialize */
  Cuparam cdp[NSPCTR];
  Model mdl[NSPCTR];
  Chain chn[NSPCTR];
  Spectrum spc[NSPCTR];

  cdp[0].dev = atoi( argv[1] );
  const char *spcFl1 = argv[2];
  const char *spcFl2 = argv[3];
  const char *spcLst[NSPCTR] = { spcFl1, spcFl2 }; //PNpwnExGrp1Real0.pi
  chn[0].thrdNm = argv[4];
  chn[0].nmbrOfWlkrs = atoi ( argv[5] );
  chn[0].nmbrOfStps = atoi ( argv[6] );
  chn[0].thrdIndx = atoi ( argv[7] );
  chn[0].dlt = dlt;
  spc[0].lwrNtcdEnrg = lwrNtcdEnrg;
  spc[0].hghrNtcdEnrg = hghrNtcdEnrg;
  spc[1].lwrNtcdEnrg = lwrNtcdEnrg;
  spc[1].hghrNtcdEnrg = hghrNtcdEnrg;

  InitializeCuda ( cdp );
  InitializeModel ( mdl );
  InitializeChain ( cdp, phbsPwrlwInt, chn );

  SpecInfo ( spcLst, spc );
  SpecAlloc ( chn, spc );
  SpecData ( cdp, verbose, mdl, spc );

  /* Initialize walkers */
  if ( chn[0].thrdIndx == 0 )
  {
    InitAtRandom ( cdp, chn );
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
      Priors ( mdl, chn );
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
  SimpleWriteDataFloat ( "Autocor.dat", chn[0].nmbrOfStps, chn[0].atCrrFnctn );
  SimpleWriteDataFloat ( "AutocorCM.dat", chn[0].nmbrOfStps, chn[0].cmSmAtCrrFnctn );
  WriteChainToFile ( chn[0].thrdNm, chn[0].thrdIndx, chn[0].nmbrOfWlkrs, chn[0].nmbrOfStps, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs );

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
