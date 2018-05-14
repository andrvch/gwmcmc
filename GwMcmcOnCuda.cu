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
  cndtn = cndtn * ( wlkr.par[indx] > 0 ) * ( wlkr.par[indx] < 5.5 );
  indx = 1; // pl normalization
  cndtn = cndtn * ( wlkr.par[indx] > -15. );
  indx = 2; // Temperature
  cndtn = cndtn * ( wlkr.par[indx] > 0.03 ) * ( wlkr.par[indx] < 1. );
  indx = 3; // Radi
  //cndtn = cndtn * ( wlkr.par[indx] > 0.0 );
  indx = 4; // Distance
  cndtn = cndtn * ( wlkr.par[indx] < 3.3 ) * ( wlkr.par[indx] > 1. );
  indx = NHINDX; // Hydrogen column density
  cndtn = cndtn * ( wlkr.par[indx] > 0 );
  return cndtn;
}

__host__ __device__ float PriorStatistic ( const Walker wlkr, const int cndtn, const float mNh, const float sNh )
{
  int indx = NHINDX; // Hydrogen column density
  float prr = 0, sum = 0, mean = 0, sigma = 0.06;
  float theta = powf ( sNh, 2 ) / mNh;
  float kk = mNh / theta;
  sum = sum + ( kk - 1 ) * logf ( wlkr.par[indx] ) - wlkr.par[indx] / theta;
  //sum = sum + powf ( ( wlkr.par[indx] - mNh ) / sNh, 2 );
  indx = NHINDX + 1;
  while ( indx < NPRS )
  {
    sum = sum + powf ( ( wlkr.par[indx] - mean ) / sigma, 2 );
    indx += 1;
  }
  if ( cndtn ) { prr = sum; } else { prr = INF; }
  return prr;
}

__global__ void AssembleArrayOfAbsorptionFactors ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int nmbrOfElmnts,
                                                   const float *crssctns, const float *abndncs, const int *atmcNmbrs, const Walker *wlkrs, float *absrptnFctrs )
{
  int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
  int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
  int ttIndx = enIndx + wlIndx * nmbrOfEnrgChnnls;
  int elIndx, effElIndx, crIndx, prIndx;
  float xsctn, clmn, nh;
  if ( ( enIndx < nmbrOfEnrgChnnls ) && ( wlIndx < nmbrOfWlkrs ) )
  {
    elIndx = 0;
    prIndx = elIndx + NHINDX;
    crIndx = elIndx + enIndx * nmbrOfElmnts;
    effElIndx = atmcNmbrs[elIndx] - 1;
    nh = wlkrs[wlIndx].par[prIndx] * 1.E22;
    clmn = abndncs[effElIndx];
    xsctn = clmn * crssctns[crIndx];
    elIndx = 1;
    while ( elIndx < nmbrOfElmnts )
    {
      prIndx = elIndx + NHINDX;
      crIndx = elIndx + enIndx * nmbrOfElmnts;
      effElIndx = atmcNmbrs[elIndx] - 1;
      clmn = abndncs[effElIndx]; // * powf ( 10, wlkrs[wlIndx].par[prIndx] );
      xsctn = xsctn + clmn * crssctns[crIndx];
      elIndx += 1;
    }
    absrptnFctrs[ttIndx] = expf ( - nh * xsctn );
  }
}

__global__ void AssembleArrayOfModelFluxes ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls,
                                             const float *enrgChnnls, const float *arfFctrs, const float *absrptnFctrs, const Walker *wlkrs,
                                             float *mdlFlxs )
{
  int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
  int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
  int ttIndx = enIndx + wlIndx * nmbrOfEnrgChnnls;
  if ( ( enIndx < nmbrOfEnrgChnnls ) && ( wlIndx < nmbrOfWlkrs ) )
  {
    //mdlFlxs[ttIndx] =  arfFctrs[enIndx] * absrptnFctrs[ttIndx] * BlackBody ( wlkrs[wlIndx].par[0], wlkrs[wlIndx].par[1], enrgChnnls[enIndx], enrgChnnls[enIndx+1] );
    //mdlFlxs[ttIndx] =  arfFctrs[enIndx] * absrptnFctrs[ttIndx] * PowerLaw ( wlkrs[wlIndx].par[0], wlkrs[wlIndx].par[1], enrgChnnls[enIndx], enrgChnnls[enIndx+1] );
    mdlFlxs[ttIndx] =  arfFctrs[enIndx] * absrptnFctrs[ttIndx] * ( PowerLaw ( wlkrs[wlIndx].par[0], wlkrs[wlIndx].par[1], enrgChnnls[enIndx], enrgChnnls[enIndx+1] ) + BlackBody ( wlkrs[wlIndx].par[2], wlkrs[wlIndx].par[3], enrgChnnls[enIndx], enrgChnnls[enIndx+1] ) );
  }
}

__host__ int ModelFluxes ( const float *abndncs, const int *atmcNmbrs, const int nmbrOfWlkrs, const Walker *wlkrs, const int nmbrOfEnrgChnnls, const float *crssctns, const float *enrgChnnls, const float *arfFctrs, float *absrptnFctrs, float *mdlFlxs )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  dim3 dimGrid = Grid ( nmbrOfEnrgChnnls, nmbrOfWlkrs );
  AssembleArrayOfAbsorptionFactors <<< dimGrid, dimBlock >>> ( nmbrOfWlkrs, nmbrOfEnrgChnnls, ATNMR, crssctns, abndncs, atmcNmbrs, wlkrs, absrptnFctrs );
  /* 4 a ) Assemble array of nsa fluxes */
  //BilinearInterpolation <<< dimGrid_0, dimBlock >>> ( chn[0].nmbrOfWlkrs, spc[0].nmbrOfEnrgChnnls, 2, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spc[0].enrgChnnls, chn[0].wlkrs, spc[0].mdlFlxs );
  /* 4 ) Assemble array of model fluxes, spc[0].mdlFlxs[chn[0].nmbrOfWlkrs*spc[0].nmbrOfEnrgChnnls] */
  AssembleArrayOfModelFluxes <<< dimGrid, dimBlock >>> ( nmbrOfWlkrs, nmbrOfEnrgChnnls, enrgChnnls, arfFctrs, absrptnFctrs, wlkrs, mdlFlxs );
  return 0;
}

__host__ int Priors ( const int nmbrOfDistBins, const float *Dist, const float *EBV, const float *errEBV, const int nmbrOfWlkrs, Walker *prpsdWlkrs, float *mNh, float *sNh, float *prrs )
{
  int blocksPerThread = Blocks ( nmbrOfWlkrs );
  LinearInterpolation <<< blocksPerThread, THRDSPERBLCK >>> ( nmbrOfWlkrs, nmbrOfDistBins, 4, Dist, EBV, errEBV, prpsdWlkrs, mNh, sNh );
  AssembleArrayOfPriors <<< blocksPerThread, THRDSPERBLCK >>> ( nmbrOfWlkrs, prpsdWlkrs, mNh, sNh, prrs );
  return 0;
}

/**
 * Host main routine
 */
int main ( int argc, char *argv[] )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  const int verbose = 1;
  const float lwrNtcdEnrg = 0.3;
  const float hghrNtcdEnrg = 8.0;
  const float dlt = 1.E-4;
  const float phbsPwrlwInt[NPRS] = { 1.1, log10f ( 9.E-6 ), 0.1, -3., log10f ( 8E2 ), 0.15 };

  /* Initialize */
  Cuparam cdp[NSPCTR];
  Model mdl[NSPCTR];
  Chain chn[NSPCTR];
  Spectrum spc[NSPCTR];

  cdp[0].dev = atoi( argv[1] );
  const char *spcFl = argv[2];
  const char *spcLst[NSPCTR] = { spcFl, spcFl };
  chn[0].thrdNm = argv[3];
  chn[0].nmbrOfWlkrs = atoi ( argv[4] );
  chn[0].nmbrOfStps = atoi ( argv[5] );
  chn[0].thrdIndx = atoi ( argv[6] );
  chn[0].dlt = dlt;
  spc[0].lwrNtcdEnrg = lwrNtcdEnrg;
  spc[0].hghrNtcdEnrg = hghrNtcdEnrg;
  spc[1].lwrNtcdEnrg = lwrNtcdEnrg;
  spc[1].hghrNtcdEnrg = hghrNtcdEnrg;

  InitializeCuda ( cdp );
  InitializeModel ( mdl );
  InitializeChain ( cdp, phbsPwrlwInt, chn );
  InitializeSpectra ( spcLst, cdp, verbose, chn, mdl, spc );

  /* Initialize walkers */
  if ( chn[0].thrdIndx == 0 )
  {
    InitializeWalkers ( cdp, chn[0].nmbrOfWlkrs, chn[0].rndmVls, chn[0].dlt, chn[0].strtngWlkr, chn[0].wlkrs, chn[0].sttstcs );

    ModelFluxes ( mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].nmbrOfWlkrs, chn[0].wlkrs, spc[0].nmbrOfEnrgChnnls, spc[0].crssctns, spc[0].enrgChnnls, spc[0].arfFctrs, spc[0].absrptnFctrs, spc[0].mdlFlxs );
    FoldModelFluxes ( cdp, chn[0].nmbrOfWlkrs, spc[0].nmbrOfChnnls, spc[0].nmbrOfEnrgChnnls, spc[0].nmbrOfRmfVls, spc[0].rmfVls, spc[0].rmfPntr, spc[0].rmfIndx, spc[0].mdlFlxs, spc[0].flddMdlFlxs );
    Statistics ( chn[0].nmbrOfWlkrs, spc[0].nmbrOfChnnls, spc[0].srcExptm, spc[0].bckgrndExptm, spc[0].srcCnts, spc[0].bckgrndCnts, spc[0].flddMdlFlxs, spc[0].chnnlSttstcs );
    SumUpStatistics ( cdp, chn[0].nmbrOfWlkrs, spc[0].nmbrOfChnnls, spc[0].chnnlSttstcs, spc[0].ntcdChnnls, chn[0].sttstcs );
  }
  else if ( chn[0].thrdIndx > 0 )
  {
    InitializeWalkersAndStatistics ( chn[0].nmbrOfWlkrs, chn[0].lstWlkrsAndSttstcs, chn[0].wlkrs, chn[0].sttstcs );
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
      Propose ( stpIndx, sbstIndx, chn[0].nmbrOfWlkrs / 2, chn[0].wlkrs, chn[0].rndmVls, chn[0].zRndmVls, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs );

      Priors ( mdl[0].nmbrOfDistBins, mdl[0].Dist, mdl[0].EBV, mdl[0].errEBV, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh, chn[0].prrs );

      ModelFluxes ( mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, spc[0].nmbrOfEnrgChnnls, spc[0].crssctns, spc[0].enrgChnnls, spc[0].arfFctrs, spc[0].absrptnFctrs, spc[0].mdlFlxs );
      FoldModelFluxes ( cdp, chn[0].nmbrOfWlkrs / 2, spc[0].nmbrOfChnnls, spc[0].nmbrOfEnrgChnnls, spc[0].nmbrOfRmfVls, spc[0].rmfVls, spc[0].rmfPntr, spc[0].rmfIndx, spc[0].mdlFlxs, spc[0].flddMdlFlxs );
      Statistics ( chn[0].nmbrOfWlkrs / 2, spc[0].nmbrOfChnnls, spc[0].srcExptm, spc[0].bckgrndExptm, spc[0].srcCnts, spc[0].bckgrndCnts, spc[0].flddMdlFlxs, spc[0].chnnlSttstcs );
      SumUpStatistics ( cdp, chn[0].nmbrOfWlkrs / 2, spc[0].nmbrOfChnnls, spc[0].chnnlSttstcs, spc[0].ntcdChnnls, chn[0].prpsdSttstcs );

      Update ( stpIndx, sbstIndx, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs, chn[0].prrs, chn[0].rndmVls, chn[0].zRndmVls, chn[0].wlkrs, chn[0].sttstcs );
      sbstIndx += 1;
    }
    WriteWalkersAndStatisticsToChain <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, stpIndx, chn[0].wlkrs, chn[0].sttstcs, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs );
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
