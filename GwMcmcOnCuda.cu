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
  float frq, phs;
  frq = wlkr.par[0];
  //cndtn = cndtn * ( 3.32 < frq ) * ( frq < 3.40 );
  cndtn = cndtn * ( 2.39 < frq ) * ( frq < 2.44 );
  phs = wlkr.par[1];
  cndtn = cndtn * ( 0.0 < phs ) * ( phs < 1. );
  /*for ( int i = FIRSTBIN; i <  NPRS; i++ )
  {
    cndtn = cndtn * ( 0. < wlkr.par[i] );
  }*/
  return cndtn;
}

__host__ __device__ float PriorStatistic ( const Walker wlkr, const int cndtn )
{
  float prr = 0, sum = 2. * logf ( 2 * wlkr.par[0]);
  if ( cndtn ) { prr = sum; } else { prr = INF; }
  return prr;
}

__global__ void AssembleArrayOfPriors ( const int nmbrOfWlkrs, const Walker *wlkrs, float *prrs )
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nmbrOfWlkrs )
  {
    prrs[i] = PriorStatistic ( wlkrs[i], PriorCondition ( wlkrs[i] ) );
  }
}

__host__ int Priors ( const Model *mdl, const int nmbrOfWlkrs, const Walker *wlkrs, float *prrs )
{
  AssembleArrayOfPriors <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, wlkrs, prrs );
  return 0;
}

__host__ __device__ float GregoryLoredo ( const float tms, const Walker wlkr, const float Ttot, const int N )
{
  float sttstc = 0, frq, phs, jt, jtFr, jtJt, jtInt, A;
  int jIndx;
  frq = wlkr.par[0]; // * 1.E-6 + F0;
  phs = wlkr.par[1];
  jt = 1 + NTBINS * fmodf ( 2 * PI * ( frq * tms + phs ), 2 * PI ) / 2 / PI;
  jtFr = modff( jt, &jtInt );
  jtJt = jt - jtFr;
  jIndx = llroundf ( jtJt );
  A = SumOfComponents ( wlkr ) / NTBINS;
  sttstc = logf ( NTBINS * A ) - A * Ttot / N + logf ( wlkr.par[jIndx+FIRSTBIN-1] / A / NTBINS );
  return sttstc;
}

__host__ __device__ int BinNumber ( const float tms, const Walker wlkr )
{
  float frq, phs, jt, jtFr, jtJt, jtInt;
  int jIndx;
  frq = wlkr.par[0]; // * 1.E-6 + F0;
  phs = wlkr.par[1];
  jt = 1 + NTBINS * fmodf ( 2 * PI * ( frq * tms + phs ), 2 * PI ) / 2 / PI;
  jtFr = modff( jt, &jtInt );
  jtJt = jt - jtFr;
  jIndx = llroundf ( jtJt );
  return jIndx;
}

__global__ void AssembleArrayOfTimesStatistic ( const int nmbrOfWlkrs, const int nmbrOfPhtns, const float srcExptm, const Walker *wlk, const float *arrTms, float *tmsSttstcs )
{
  int a = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t = a + w * nmbrOfPhtns;
  if ( ( a < nmbrOfPhtns ) && ( w < nmbrOfWlkrs ) )
  {
    tmsSttstcs[t] = - 2. * GregoryLoredo ( arrTms[a], wlk[w], srcExptm, nmbrOfPhtns );
  }
}

__global__ void AssembleArrayOfBinTimes ( const int nmbrOfWlkrs, const int nmbrOfPhtns, const Walker *wlk, const float *arrTms, float *nntms )
{
  int a = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t;
  if ( ( a < nmbrOfPhtns ) && ( w < nmbrOfWlkrs ) )
  {
    for ( int j = 0; j < NTBINS; j++ )
    {
      t = a + j * nmbrOfPhtns + w * nmbrOfPhtns * NTBINS ;
      if ( j+1 == BinNumber ( arrTms[a], wlk[w] ) )
      {
        nntms[t] = 1.;
      }
      else
      {
        nntms[t] = 0.;
      }
    }
  }
}

__host__ int StatTimes ( const int nmbrOfWlkrs, const Walker *wlk, Spectrum spec )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  dim3 dimGrid = Grid ( spec.nmbrOfPhtns, nmbrOfWlkrs );
  //AssembleArrayOfTimesStatistic <<< dimGrid, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfPhtns, spec.srcExptm, wlk, spec.arrTms, spec.tmsSttstcs );
  AssembleArrayOfBinTimes <<< dimGrid, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfPhtns, wlk, spec.arrTms, spec.nnTms );
  return 0;
}

__global__ void AssembleArrayOfMultiplicity ( const int nmbrOfWlkrs, const int nmbrOfPhtns, const float Ttot, const float *nTms, float *sttstcs )
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float sum;
  if ( i < nmbrOfWlkrs )
  {
    sum = 0;
    for ( int b = 0; b < NTBINS; b++ )
    {
      sum += nTms[b+i*NTBINS] * logf ( nTms[b+i*NTBINS] / nmbrOfPhtns / Ttot ) - nTms[b+i*NTBINS] + 0.5 * logf ( nTms[b+i*NTBINS] / Ttot );
    }
    sttstcs[i] = - 2. * sum;
  }
}

__host__ int SumUpStat ( Cuparam *cdp, const float beta, const int nmbrOfWlkrs, float *nTms, float *sttstcs, const Spectrum spec )
{
  float alpha = ALPHA;
  cdp[0].cublasStat = cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spec.nmbrOfPhtns, nmbrOfWlkrs * NTBINS, &alpha, spec.nnTms, spec.nmbrOfPhtns, spec.ntcdTms, INCXX, &beta, nTms, INCYY );
  if ( cdp[0].cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed 0 " ); return 1; }
  AssembleArrayOfMultiplicity <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, spec.nmbrOfPhtns, spec.srcExptm, nTms, sttstcs );
  return 0;
}

__host__ int ReadTimesInfo ( const char *spcFl, int *nmbrOfPhtns, float *srcExptm )
{
  fitsfile *fptr;      /* FITS file pointer, defined in fitsio.h */
  int status = 0, hdutype;   /*  CFITSIO status value MUST be initialized to zero!  */
  long nrows;
  fits_open_file ( &fptr, spcFl, READONLY, &status );
  fits_movabs_hdu ( fptr, 2, &hdutype, &status );
  fits_get_num_rows ( fptr, &nrows, &status );
  printf ( "%i\n", status );
  *nmbrOfPhtns = nrows;
  printf ( "%i\n", *nmbrOfPhtns );
  //snprintf ( card, sizeof ( card ), "%s%s", spcFl, "[EVENTS]" );
  //fits_open_file ( &fptr, card, READONLY, &status );
  fits_read_key ( fptr, TFLOAT, "DURATION", srcExptm, NULL, &status );
  printf ( "%i\n", status );
  printf ( "%.8E\n", *srcExptm );
  return 0;
}

__host__ int TimesInfo ( const char *spcLst[NSPCTR], const int verbose, Spectrum *spc )
{
  for ( int i = 0; i < NSPCTR; i++ )
  {
    ReadTimesInfo ( spcLst[i], &spc[i].nmbrOfPhtns, &spc[i].srcExptm );
  }
  return 0;
}

__host__ int TimesAlloc ( Chain *chn, Spectrum *spc )
{
  for ( int i = 0; i < NSPCTR; i++ )
  {
    cudaMallocManaged ( ( void ** ) &spc[i].ntcdTms, spc[i].nmbrOfPhtns * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].tmsSttstcs, spc[i].nmbrOfPhtns * chn[0].nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].arrTms, spc[i].nmbrOfPhtns * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].nnTms, spc[i].nmbrOfPhtns * chn[0].nmbrOfWlkrs * NTBINS * sizeof ( float ) );
  }
  return 0;
}

__global__ void AssembleArrayOfNoticedTimes ( const int nmbrOfPhtns, float *ntcdTms )
{
  int a = threadIdx.x + blockDim.x * blockIdx.x;
  if ( a < nmbrOfPhtns )
  {
    ntcdTms[a] = 1.;
  }
}

__host__ int TimesData ( const char *spcFl[NSPCTR], Cuparam *cdp, const int verbose, Spectrum *spc )
{
  for ( int i = 0; i < NSPCTR; i++ )
  {
    ReadTimesData ( verbose, spcFl[i], spc[i].nmbrOfPhtns, spc[i].arrTms );
    AssembleArrayOfNoticedTimes <<< Blocks ( spc[i].nmbrOfPhtns ), THRDSPERBLCK >>> ( spc[i].nmbrOfPhtns, spc[i].ntcdTms );
  }
  return 0;
}

__host__ int ReadTimesData ( const int verbose, const char *spcFl, const int nmbrOfPhtns, float *arrTms )
{
  fitsfile *fptr;      /* FITS file pointer, defined in fitsio.h */
  int status = 0, hdutype;   /*  CFITSIO status value MUST be initialized to zero!  */
  long nrows;
  long  firstrow=1, firstelem = 1;
  int colnum = 1, anynul;
  float enullval = 0.0;
  fits_open_file ( &fptr, spcFl, READONLY, &status );
  fits_movabs_hdu ( fptr, 2, &hdutype, &status );
  fits_get_num_rows ( fptr, &nrows, &status );
  int numData = nrows;
  double *tms0;
  cudaMallocManaged ( ( void ** ) &tms0, numData * sizeof ( double ) );
  fits_read_col_dbl ( fptr, colnum, firstrow, firstelem, nrows, enullval, tms0, &anynul, &status );
  fits_close_file( fptr, &status );
  for ( int i = 0; i < nrows; i++ )
  {
    arrTms[i] = tms0[i] - tms0[0];
  }
  cudaFree ( tms0 );
  return 0;
}

/**
 * Host main routine
 */
int main ( int argc, char *argv[] )
{
  dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
  const int verbose = 1;
  const float lwrNtcdEnrg1 = 0.;
  const float hghrNtcdEnrg1 = 12.0;
  const float dlt = 1.E-5;
  const float phbsPwrlwInt[NPRS] = { 2.41720983, 0.5 };

  /* Initialize */
  Cuparam cdp[NSPCTR];
  Model mdl[NSPCTR];
  Chain chn[NSPCTR];
  Spectrum spc[NSPCTR];

  cdp[0].dev = atoi( argv[1] );
  const char *spcFl1 = argv[2];
  const char *tmsLst[NSPCTR] = { spcFl1 };

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

  TimesInfo ( tmsLst, verbose, spc );
  TimesAlloc ( chn, spc );
  TimesData ( tmsLst, cdp, verbose, spc );

  /* Initialize walkers */
  if ( chn[0].thrdIndx == 0 )
  {
    InitAtRandom ( cdp, chn );
    Priors ( mdl, chn[0].nmbrOfWlkrs, chn[0].wlkrs, chn[0].prrs );
    for ( int i = 0; i < NSPCTR; i++ )
    {
      StatTimes ( chn[0].nmbrOfWlkrs, chn[0].wlkrs, spc[i] );
      SumUpStat ( cdp, 1, chn[0].nmbrOfWlkrs, chn[0].nTms, chn[0].sttstcs, spc[i] );
      cudaDeviceSynchronize ();
      for ( int k = 0; k < chn[0].nmbrOfWlkrs; k++ )
      {
        printf ( " %3.9f ", chn[0].rndmVls[2*k] );
        printf ( " %3.9f ", chn[0].rndmVls[2*k+1] );
        printf ( " %3.9f ", chn[0].rndmWlkr[k].par[0] );
        printf ( " %3.9f ", chn[0].rndmWlkr[k].par[1] );
        printf ( " %3.9f ", chn[0].wlkrs[k].par[0] );
        printf ( " %3.9f ", chn[0].wlkrs[k].par[1] );
        for ( int j = 0; j < NTBINS; j++ )
        {
          printf ( " %3.3f ", chn[0].nTms[j+k*NTBINS] );
        }
        printf ( " %3.9f ", chn[0].sttstcs[k] );
        printf ( "\n" );
      }
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
  curandGenerateNormal ( cdp[0].curandGnrtr,  chn[0].rndmVls1, chn[0].nmbrOfStps * chn[0].nmbrOfWlkrs, 0., 0.1 );
  curandGenerateNormal ( cdp[0].curandGnrtr,  chn[0].rndmVls2, chn[0].nmbrOfStps * chn[0].nmbrOfWlkrs, 0., 0.1 );

  AssembleArrayOfRandom2DWalkersFromTwoRandomArrays <<< Blocks ( spc[0].nmbrOfPhtns ), THRDSPERBLCK >>> ( const int n, const float *a, const float *b, Walker *w )

  int stpIndx = 0, sbstIndx;
  while ( stpIndx < chn[0].nmbrOfStps )
  {
    sbstIndx = 0;
    while ( sbstIndx < 2 )
    {
      Propose ( stpIndx, sbstIndx, chn );
      Priors ( mdl, chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].prpsdPrrs );
      for ( int i = 0; i < NSPCTR; i++ )
      {
        StatTimes ( chn[0].nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, spc[i] );
        SumUpStat ( cdp, 1, chn[0].nmbrOfWlkrs / 2, chn[0].nTms, chn[0].prpsdSttstcs, spc[i] );
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
  SimpleWriteDataFloat ( "ArrTms.out", spc[0].nmbrOfPhtns, spc[0].arrTms );
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
