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
<<<<<<< HEAD
  cndtn = cndtn * ( log10f ( 8. ) < wlkr.par[RINDX1] ) * ( wlkr.par[RINDX1] < log10f ( 20. ) );
  cndtn = cndtn * ( log10f ( 80. ) < wlkr.par[DINDX1] ) * ( wlkr.par[DINDX1] < log10f ( 2200. ) );
  cndtn = cndtn * ( log10f ( 0.7 ) < wlkr.par[3] ) * ( wlkr.par[3] < log10f ( 0.9 ) );
  cndtn = cndtn * ( log10f ( 0.01 ) < wlkr.par[4] ) * ( wlkr.par[4] < log10f ( 0.5 ) );
=======
>>>>>>> two_spectra+background
  cndtn = cndtn * ( 0. < wlkr.par[NHINDX] );
  return cndtn;
}

__host__ __device__ float PriorStatistic ( const Walker wlkr, const int cndtn, const float mNh1, const float sNh1, const float mNh2, const float sNh2 )
{
  float prr = 0, sum = 0;
  float theta = powf ( sNh1, 2 ) / mNh1;
  float kk = mNh1 / theta;
  //sum = sum + ( kk - 1 ) * logf ( wlkr.par[NHINDX] ) - wlkr.par[NHINDX] / theta;
  //sum = sum + powf ( ( wlkr.par[NHINDX] - mNh1 ) / sNh1, 2 );
  if ( cndtn ) { prr = sum; } else { prr = INF; }
  return prr;
}

__global__ void AssembleArrayOfModelFluxes ( const int spIndx, const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const float backscal_src, const float backscal_bkg, const float *en, const float *arf, const float *absrptn, const Walker *wlk, const float *nsa1Flx, const float *nsa2Flx, float *flx )
{
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t = e + w * nmbrOfEnrgChnnls;
  float f = 0, NormD, intNsaFlx;
  float scl = backscal_src / backscal_bkg;
  if ( ( e < nmbrOfEnrgChnnls ) && ( w < nmbrOfWlkrs ) )
  {
    if ( spIndx == 0 )
    {
<<<<<<< HEAD
      f = f + nsa1Flx[t]; // * powf ( 10., LOGPLANCK - log10f ( en[e+1] ) );
      f = f * GaussianAbsorption ( wlk[w].par[3], wlk[w].par[4], wlk[w].par[5], en[e+1] );
      f = f + PowerLaw ( wlk[w].par[6], wlk[w].par[7], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[8], wlk[w].par[9], en[e], en[e+1] );
=======
      NormD = powf ( 10., - 2. * ( wlk[w].par[DINDX1] - KMCMPCCM ) );
      intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      f = f + NormD * intNsaFlx;
      f = f + PowerLaw ( wlk[w].par[2], wlk[w].par[3], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[4], wlk[w].par[5], en[e], en[e+1] );
>>>>>>> two_spectra+background
      flx[t] = f * arf[e];
    }
    if ( spIndx == 1 )
    {
<<<<<<< HEAD
      f = f + PowerLaw ( wlk[w].par[8], wlk[w].par[9], en[e], en[e+1] );
=======
      f = f + PowerLaw ( wlk[w].par[4], wlk[w].par[5], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 2 )
    {
      NormD = powf ( 10., - 2. * ( wlk[w].par[DINDX1] - KMCMPCCM ) );
      intNsaFlx = NormD * IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      f = f + intNsaFlx;
      f = f + PowerLaw ( wlk[w].par[2], wlk[w].par[3], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[6], wlk[w].par[7], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 3 )
    {
      f = f + PowerLaw ( wlk[w].par[6], wlk[w].par[7], en[e], en[e+1] );
>>>>>>> two_spectra+background
      flx[t] = f * arf[e];
    }
  }
}

__host__ int ModelFluxes ( const Model *mdl, const int nmbrOfWlkrs, const Walker *wlkrs, const int indx, Spectrum spec )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  AssembleArrayOfAbsorptionFactors <<< Grid ( spec.nmbrOfEnrgChnnls, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, ATNMR, spec.crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, wlkrs, spec.absrptnFctrs );
  //BilinearInterpolation <<< Grid ( spec.nmbrOfEnrgChnnls+1, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, TINDX, RINDX1, DINDX1, mdl[0].nsmaxgFlxs, mdl[0].nsmaxgE, mdl[0].nsmaxgT, mdl[0].numNsmaxgE, mdl[0].numNsmaxgT, spec.enrgChnnls, wlkrs, spec.nsa1Flxs );
  BilinearInterpolation <<< Grid ( spec.nmbrOfEnrgChnnls+1, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls+1, TINDX, RINDX1, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spec.enrgChnnls, wlkrs, spec.nsa1Flxs );
  AssembleArrayOfModelFluxes <<< Grid ( spec.nmbrOfEnrgChnnls, nmbrOfWlkrs ), dimBlock >>> ( indx, nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, spec.backscal_src, spec.backscal_bkg, spec.enrgChnnls, spec.arfFctrs, spec.absrptnFctrs, wlkrs, spec.nsa1Flxs, spec.nsa2Flxs, spec.mdlFlxs );
  return 0;
}

__host__ int Priors ( const Model *mdl, Chain *chn )
{
  int blcks = Blocks ( chn[0].nmbrOfWlkrs / 2 );
  LinearInterpolation <<< blcks, THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs / 2, mdl[0].nmbrOfDistBins, DINDX1, mdl[0].Dist, mdl[0].EBV, mdl[0].errEBV, chn[0].prpsdWlkrs, chn[0].mNh1, chn[0].sNh1 );
  AssembleArrayOfPriors <<< blcks, THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].mNh1, chn[0].sNh1, chn[0].mNh2, chn[0].sNh2, chn[0].prrs );
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
  const float hghrNtcdEnrg = 7.0;
  const float dlt = 1.E-4;
<<<<<<< HEAD
  const float phbsPwrlwInt[NPRS] = { 5.80, 1.0, 2.6, log10f ( 0.8 ), log10f ( 0.15 ), -1., 1.2, -5.2, 0.9, -5.0, 0.30 };
=======
  const float phbsPwrlwInt[NPRS] = { 6.0, 3.5, 1.0, -5.3, 0.90, -5.0, 1.2, -5.1, 0.17 };
>>>>>>> two_spectra+background

  /* Initialize */
  Cuparam cdp[NSPCTR];
  Model mdl[NSPCTR];
  Chain chn[NSPCTR];
  Spectrum spc[NSPCTR];

  cdp[0].dev = atoi( argv[1] );
  const char *spcFl1 = argv[2];
  const char *spcFl2 = argv[3];
  const char *spcFl3 = argv[4];
  const char *spcFl4 = argv[5];
  const char *spcLst[NSPCTR] = { spcFl1, spcFl2, spcFl3, spcFl4 };
  int NNspec = 4;
  chn[0].thrdNm = argv[NNspec+2];
  chn[0].nmbrOfWlkrs = atoi ( argv[NNspec+3] );
  chn[0].nmbrOfStps = atoi ( argv[NNspec+4] );
  chn[0].thrdIndx = atoi ( argv[NNspec+5] );
  chn[0].dlt = dlt;
  for ( int i = 0; i < NSPCTR; i++ )
  {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg;
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
  SimpleWriteDataFloat ( "Autocor.out", chn[0].nmbrOfStps, chn[0].atCrrFnctn );
  SimpleWriteDataFloat ( "AutocorCM.out", chn[0].nmbrOfStps, chn[0].cmSmAtCrrFnctn );
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
