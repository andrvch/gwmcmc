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

__global__ void AssembleArrayOfModelFluxes ( const int spIndx, const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const float backscal_src, const float backscal_bkg, const float *en, const float *arf, const float *absrptn, const Walker *wlk, const float *nsa1Flx, float *flx )
{
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t = e + w * nmbrOfEnrgChnnls;
  float f = 0, Norm, intNsaFlx;
  float scl = backscal_src / backscal_bkg;
  if ( ( e < nmbrOfEnrgChnnls ) && ( w < nmbrOfWlkrs ) )
  {
    if ( spIndx == 0 )
    {
      intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      //Norm = powf ( 10., 2. * ( wlk[w].par[RINDX1] - log10f ( RNS ) - wlk[w].par[DINDX1] + KMCMPCCM ) );
      Norm = powf ( 10., 2. * ( wlk[w].par[RINDX1] + KMCMPCCM ) );
      f = f + Norm * intNsaFlx;
      f = f + PowerLaw ( wlk[w].par[3], wlk[w].par[4], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[7], wlk[w].par[8], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 1 )
    {
      f = f + PowerLaw ( wlk[w].par[7], wlk[w].par[8], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 2 )
    {
      intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      //Norm = powf ( 10., 2. * ( wlk[w].par[RINDX1] - log10f ( RNS ) - wlk[w].par[DINDX1] + KMCMPCCM ) );
      Norm = powf ( 10., 2. * ( wlk[w].par[RINDX1] + KMCMPCCM ) );
      f = f + Norm * intNsaFlx;
      f = f + PowerLaw ( wlk[w].par[3], wlk[w].par[4], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[9], wlk[w].par[10], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 3 )
    {
      f = f + PowerLaw ( wlk[w].par[9], wlk[w].par[10], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 4 )
    {
      intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      //Norm = powf ( 10., 2. * ( wlk[w].par[RINDX1] - log10f ( RNS ) - wlk[w].par[DINDX1] + KMCMPCCM ) );
      Norm = powf ( 10., 2. * ( wlk[w].par[RINDX1] + KMCMPCCM ) );
      f = f + Norm * intNsaFlx;
      f = f + PowerLaw ( wlk[w].par[3], wlk[w].par[4], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[11], wlk[w].par[12], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 5 )
    {
      f = f + PowerLaw ( wlk[w].par[11], wlk[w].par[12], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 6 )
    {
      f = f + PowerLaw ( wlk[w].par[5], wlk[w].par[6], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[7], wlk[w].par[8], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 7 )
    {
      f = f + PowerLaw ( wlk[w].par[7], wlk[w].par[8], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 8 )
    {
      f = f + PowerLaw ( wlk[w].par[5], wlk[w].par[6], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[9], wlk[w].par[10], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 9 )
    {
      f = f + PowerLaw ( wlk[w].par[9], wlk[w].par[10], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 10 )
    {
      f = f + PowerLaw ( wlk[w].par[5], wlk[w].par[6], en[e], en[e+1] );
      f = f * absrptn[t];
      f = f + scl * PowerLaw ( wlk[w].par[11], wlk[w].par[12], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
    if ( spIndx == 11 )
    {
      f = f + PowerLaw ( wlk[w].par[11], wlk[w].par[12], en[e], en[e+1] );
      flx[t] = f * arf[e];
    }
  }
}

__host__ int ModelFluxes ( const Model *mdl, const int nmbrOfWlkrs, const Walker *wlkrs, const int indx, Spectrum spec )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  AssembleArrayOfAbsorptionFactors <<< Grid ( spec.nmbrOfEnrgChnnls, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, ATNMR, spec.crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, wlkrs, spec.absrptnFctrs );
  BilinearInterpolationNsmax <<< Grid ( spec.nmbrOfEnrgChnnls+1, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls+1, TINDX, GRINDX, mdl[0].nsmaxgFlxs, mdl[0].nsmaxgE, mdl[0].nsmaxgT, mdl[0].numNsmaxgE, mdl[0].numNsmaxgT, spec.enrgChnnls, wlkrs, spec.nsa1Flxs );
  //BilinearInterpolation <<< Grid ( spec.nmbrOfEnrgChnnls+1, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls+1, TINDX, GRINDX, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spec.enrgChnnls, wlkrs, spec.nsa1Flxs );
  AssembleArrayOfModelFluxes <<< Grid ( spec.nmbrOfEnrgChnnls, nmbrOfWlkrs ), dimBlock >>> ( indx, nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, spec.backscal_src, spec.backscal_bkg, spec.enrgChnnls, spec.arfFctrs, spec.absrptnFctrs, wlkrs, spec.nsa1Flxs, spec.mdlFlxs );
  return 0;
}

__host__ int Priors ( const Model *mdl, const int nmbrOfWlkrs, const Walker *wlkrs, float *nhMd, float *nhSg, float *prrs )
{
  //LinearInterpolation <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, mdl[0].nmbrOfDistBins, DINDX1, mdl[0].Dist, mdl[0].EBV, mdl[0].errEBV, wlkrs, nhMd, nhSg );
  LinearInterpolationNoErrors <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist1, mdl[0].EBV1, wlkrs, nhMd, nhSg );
  AssembleArrayOfPriors <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, wlkrs, nhMd, nhSg, prrs );
  return 0;
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
  const float lwrNtcdEnrg2 = 0.2;
  const float hghrNtcdEnrg2 = 7.0;
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
  const char *spcFl3 = argv[4];
  const char *spcFl4 = argv[5];
  const char *spcFl5 = argv[6];
  const char *spcFl6 = argv[7];
  const char *spcFl7 = argv[8];
  const char *spcFl8 = argv[9];
  const char *spcFl9 = argv[10];
  const char *spcFl10 = argv[11];
  const char *spcFl11 = argv[12];
  const char *spcFl12 = argv[13];
  const char *spcLst[NSPCTR] = { spcFl1, spcFl2, spcFl3, spcFl4, spcFl5, spcFl6, spcFl7, spcFl8, spcFl9, spcFl10, spcFl11, spcFl12 };
  int NNspec = 12;
  chn[0].thrdNm = argv[NNspec+2];
  chn[0].nmbrOfWlkrs = atoi ( argv[NNspec+3] );
  chn[0].nmbrOfStps = atoi ( argv[NNspec+4] );
  chn[0].thrdIndx = atoi ( argv[NNspec+5] );
  chn[0].dlt = dlt;
  for ( int i = 0; i < 2; i++ )
  {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg1;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg1;
  }
  for ( int i = 2; i < 6; i++ )
  {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg2;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg2;
  }
  for ( int i = 6; i < 8; i++ )
  {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg1;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg1;
  }
  for ( int i = 8; i < NSPCTR; i++ )
  {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg2;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg2;
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
