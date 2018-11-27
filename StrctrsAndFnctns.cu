#ifndef _STRCTRSANDFNCTNS_CU_
#define _STRCTRSANDFNCTNS_CU_

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
//
#include "StrctrsAndFnctns.cuh"

__host__ int InitializeCuda ( const int verbose, Cuparam *cdp ) {
  cudaRuntimeGetVersion ( cdp[0].runtimeVersion );
  cudaDriverGetVersion ( cdp[0].driverVersion );
  cudaSetDevice ( cdp[0].dev );
  cudaGetDevice ( &cdp[0].dev );
  cudaGetDeviceProperties ( &cdp[0].prop, cdp[0].dev );
  cdp[0].cusparseStat = cusparseCreate ( &cdp[0].cusparseHandle );
  cdp[0].cusparseStat = cusparseCreateMatDescr ( &cdp[0].MatDescr );
  cdp[0].cusparseStat = cusparseSetMatType ( cdp[0].MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
  cdp[0].cusparseStat = cusparseSetMatIndexBase ( cdp[0].MatDescr, CUSPARSE_INDEX_BASE_ZERO );
  cdp[0].cublasStat = cublasCreate ( &cdp[0].cublasHandle );
  curandCreateGenerator ( &cdp[0].curandGnrtr, CURAND_RNG_PSEUDO_DEFAULT );
  curandCreateGeneratorHost ( &cdp[0].curandGnrtrHst, CURAND_RNG_PSEUDO_DEFAULT );
  curandSetPseudoRandomGeneratorSeed ( cdp[0].curandGnrtr, 1234ULL );
  curandSetPseudoRandomGeneratorSeed ( cdp[0].curandGnrtrHst, 1234ULL );
  cudaEventCreate ( &cdp[0].start );
  cudaEventCreate ( &cdp[0].stop );
  if ( verbose == 1 ) {
    printf ( "\n" );
    printf ( ".................................................................\n" );
    printf ( " CUDA device ID: %d\n", cdp[0].dev );
    printf ( " CUDA device Name: %s\n", cdp[0].prop.name );
    printf ( " Driver API: v%d \n", cdp[0].driverVersion[0] );
    printf ( " Runtime API: v%d \n", cdp[0].runtimeVersion[0] );
  }
  return 0;
}

__host__ int InitializeChain ( const int verbose, Cuparam *cdp, const float *strtng, Chain *chn ) {
  chn[0].nmbrOfRndmVls = 3 * 2 * chn[0].nmbrOfWlkrs / 2 * chn[0].nmbrOfStps;
  cudaMallocManaged ( ( void ** ) &chn[0].wlkrs, chn[0].nmbrOfWlkrs * sizeof ( Walker ) );
  cudaMallocManaged ( ( void ** ) &chn[0].wlkc, chn[0].dimWlk * chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prpsdWlkrs, chn[0].nmbrOfWlkrs * sizeof ( Walker ) );
  cudaMallocManaged ( ( void ** ) &chn[0].wlkp, chn[0].dimWlk * chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chnOfWlkrs, chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps * sizeof ( Walker ) );
  cudaMallocManaged ( ( void ** ) &chn[0].sttstcs, chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prrs, chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prpsdSttstcs, chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prpsdPrrs, chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chnOfSttstcs, chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chnOfPrrs, chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].zRndmVls, chn[0].nmbrOfWlkrs / 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].rndmVls, chn[0].nmbrOfRndmVls * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].rndmVls1, chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].rndmVls2, chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].rndmWlkr, chn[0].nmbrOfWlkrs * sizeof ( Walker ) );
  cudaMallocManaged ( ( void ** ) &chn[0].rndmWlkrs1, chn[0].nmbrOfWlkrs * chn[0].nmbrOfStps * sizeof ( Walker ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chnFnctn, chn[0].nmbrOfStps * chn[0].nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].atCrrFnctn, chn[0].nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cmSmAtCrrFnctn, chn[0].nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].lstWlkrsAndSttstcs, ( NPRS + 2 ) * chn[0].nmbrOfWlkrs * sizeof ( float ) );
  if ( chn[0].thrdIndx > 0 ) {
    ReadLastPositionOfWalkersFromFile ( chn[0].thrdNm, chn[0].thrdIndx-1, chn[0].nmbrOfWlkrs, chn[0].lstWlkrsAndSttstcs );
  } else {
    for ( int i = 0; i < NPRS; i++ ) { chn[0].strtngWlkr.par[i] = strtng[i]; }
    if ( not PriorCondition ( chn[0].strtngWlkr ) ) { printf ( " !!!Initial walker doesn't satisfy prior conditions!!!\n" ); }
    if ( verbose == 1 ) {
      printf ( ".................................................................\n" );
      printf ( " Initial parameters -- " );
      for ( int i = 0; i < NPRS; i++ ) { printf ( " %2.2f ", chn[0].strtngWlkr.par[i] ); }
      printf ( "\n" );
    }
  }
  return 0;
}

__host__ int InitAtRandom ( Chain *chn ) {
  AssembleArrayOfRandomWalkers <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, chn[0].rndmVls, chn[0].rndmWlkr );
  InitializeWalkersAtRandom <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, chn[0].dlt, chn[0].strtngWlkr, chn[0].rndmWlkr, chn[0].wlkrs, chn[0].sttstcs );
  return 0;
}

__host__ int ToChain ( const int ist, Chain *chn ) {
  WriteWalkersAndStatisticsToChain <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, ist, chn[0].wlkrs, chn[0].sttstcs, chn[0].prrs, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs, chn[0].chnOfPrrs );
  return 0;
}

__host__ int Update ( const int ist, const int isb, Chain *chn ) {
  UpdateWalkers <<< Blocks ( chn[0].nmbrOfWlkrs / 2 ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs / 2, ist, isb, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs, chn[0].prpsdPrrs, chn[0].zRndmVls, chn[0].rndmVls, chn[0].wlkrs, chn[0].sttstcs, chn[0].prrs );
  return 0;
}

__host__ int MetropolisUpdate ( const int stpIndx, Chain *chn ) {
  MetropolisUpdateOfWalkers <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, stpIndx, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs, chn[0].prpsdPrrs, chn[0].rndmVls, chn[0].wlkrs, chn[0].sttstcs, chn[0].prrs );
  return 0;
}

__host__ int Propose ( const int stpIndx, const int sbstIndx, Chain *chn ) {
  GenerateProposal <<< Blocks ( chn[0].nmbrOfWlkrs / 2 ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs / 2, stpIndx, sbstIndx, chn[0].wlkrs, chn[0].rndmVls, chn[0].zRndmVls, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs );
  return 0;
}

__host__ int MetropolisPropose ( const int stpIndx, const int prmtrIndx, Chain *chn ) {
  GenerateMetropolis <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, stpIndx, prmtrIndx, chn[0].wlkrs, chn[0].rndmWlkrs1, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs );
  return 0;
}

__host__ int InitFromLast ( Chain *chn ) {
  InitializeWalkersAndStatisticsFromLastChain <<< Blocks ( chn[0].nmbrOfWlkrs ), THRDSPERBLCK >>> ( chn[0].nmbrOfWlkrs, chn[0].lstWlkrsAndSttstcs, chn[0].wlkrs, chn[0].sttstcs, chn[0].prrs );
  return 0;
}

__host__ void DestroyAllTheCudaStaff ( const Cuparam *cdp ) {
  cusparseDestroy ( cdp[0].cusparseHandle );
  cublasDestroy ( cdp[0].cublasHandle );
  curandDestroyGenerator ( cdp[0].curandGnrtr );
  curandDestroyGenerator ( cdp[0].curandGnrtrHst );
  cudaEventDestroy ( cdp[0].start );
  cudaEventDestroy ( cdp[0].stop );
  cufftDestroy ( cdp[0].cufftPlan );
}

__host__ void FreeChain ( const Chain *chn ) {
  cudaFree ( chn[0].wlkc );
  cudaFree ( chn[0].wlkp );
  cudaFree ( chn[0].wlkrs );
  cudaFree ( chn[0].prpsdWlkrs );
  cudaFree ( chn[0].chnOfWlkrs );
  cudaFree ( chn[0].sttstcs );
  cudaFree ( chn[0].prrs );
  cudaFree ( chn[0].prpsdSttstcs );
  cudaFree ( chn[0].prpsdPrrs );
  cudaFree ( chn[0].zRndmVls );
  cudaFree ( chn[0].chnOfSttstcs );
  cudaFree ( chn[0].chnOfPrrs );
  cudaFree ( chn[0].rndmVls );
  cudaFree ( chn[0].rndmVls1 );
  cudaFree ( chn[0].rndmVls2 );
  cudaFree ( chn[0].rndmWlkr );
  cudaFree ( chn[0].rndmWlkrs1 );
  cudaFree ( chn[0].chnFnctn );
  cudaFree ( chn[0].atCrrFnctn );
  cudaFree ( chn[0].cmSmAtCrrFnctn );
  cudaFree ( chn[0].lstWlkrsAndSttstcs );
}

__host__ int Blocks ( const int n ) {
  int b = ( n + THRDSPERBLCK - 1 ) / THRDSPERBLCK;
  return b;
}

__host__ dim3 Grid ( const int n, const int m ) {
  dim3 dimGrid ( ( n + THRDSPERBLCK - 1 ) / THRDSPERBLCK, ( m + THRDSPERBLCK - 1 ) / THRDSPERBLCK );
  return dimGrid;
}

__host__ __device__ Walker AddWalkers ( Walker a, Walker b ) {
  Walker c;
  for ( int i = 0; i < NPRS; i++ ) {
    c.par[i] = a.par[i] + b.par[i];
  }
  return c;
}

__host__ __device__ float SumOfComponents ( const Walker wlkr ) {
  float sum = 0;
  for ( int i = 0; i < NPRS; i++ ) {
    sum += wlkr.par[i];
  }
  return sum;
}

__host__ __device__ Walker ScaleWalker ( Walker a, float s ) {
  Walker c;
  for ( int i = 0; i < NPRS; i++ ) {
    c.par[i] = s * a.par[i];
  }
  return c;
}

__host__ __device__ Complex AddComplex ( Complex a, Complex b ) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

__host__ __device__ Complex ScaleComplex ( Complex a, float s ) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

__host__ __device__ Complex MultiplyComplex ( Complex a, Complex b ) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__host__ __device__ Complex ConjugateComplex ( Complex a ) {
  Complex c;
  c.x = a.x;
  c.y = - a.y;
  return c;
}

__host__ __device__ int FindElementIndex ( const float *a, const int n, const float x ) {
  int ju, jm, jl, jres;
  jl = 0;
  ju = n;
  while ( ju - jl > 1 ) {
    jm = floorf ( 0.5 * ( ju + jl ) );
    if ( x >= a[jm] ) {
      jl = jm;
    } else {
      ju = jm;
    }
  }
  jres = jl;
  if ( x == a[0] ) jres = 0;
  if ( x >= a[n-1] ) jres = n - 1;
  return jres;
}

__host__ void ReadLastPositionOfWalkersFromFile ( const char *thrdNm, const int indx, const int nmbrOfWlkrs, float *lstChn ) {
  FILE *fptr;
  char fl[FLEN_CARD];
  float value;
  int i = 0, k = 0, j;
  snprintf ( fl, sizeof ( fl ), "%s%i%s", thrdNm, indx, ".chain" );
  fptr = fopen ( fl, "r" );
  while ( fscanf ( fptr, "%e", &value ) == 1 ) {
    i += 1;
  }
  fclose ( fptr );
  fptr = fopen ( fl, "r" );
  while ( fscanf ( fptr, "%e", &value ) == 1 ) {
    if ( k >= i - nmbrOfWlkrs * ( NPRS + 2 ) ) {
      j = k - ( i - nmbrOfWlkrs * ( NPRS + 2 ) );
      lstChn[j] = value;
    }
    k += 1;
  }
  fclose ( fptr );
}

__host__ void WriteChainToFile ( const char *thrdNm, const int indx, const int nmbrOfWlkrs, const int nmbrOfStps, const Walker *chnOfWlkrs, const float *chnOfSttstcs, const float *chnOfPrrs ) {
  FILE *flPntr;
  char flNm[FLEN_CARD];
  int ttlChnIndx, stpIndx, wlkrIndx, prmtrIndx;
  snprintf ( flNm, sizeof ( flNm ), "%s%i%s", thrdNm, indx, ".chain" );
  flPntr = fopen ( flNm, "w" );
  stpIndx = 0;
  while ( stpIndx < nmbrOfStps ) {
    wlkrIndx = 0;
    while ( wlkrIndx < nmbrOfWlkrs ) {
      ttlChnIndx = wlkrIndx + stpIndx * nmbrOfWlkrs;
      prmtrIndx = 0;
      while ( prmtrIndx < NPRS ) {
        fprintf ( flPntr, " %.8E ", chnOfWlkrs[ttlChnIndx].par[prmtrIndx] );
        prmtrIndx += 1;
      }
      fprintf ( flPntr, " %.8E ", chnOfSttstcs[ttlChnIndx] );
      prmtrIndx += 1;
      fprintf ( flPntr, " %.8E\n", chnOfPrrs[ttlChnIndx] );
      wlkrIndx += 1;
    }
    stpIndx += 1;
  }
  fclose ( flPntr );
}

__host__ void SimpleReadDataFloat ( const char *fl, float *data ) {
  FILE *fptr;
  fptr = fopen ( fl, "r" );
  float value;
  int i = 0;
  while ( fscanf ( fptr, "%e", &value ) == 1 ) {
    data[i] = value;
    i += 1;
  }
  fclose ( fptr );
}

__host__ void SimpleReadDataInt ( const char *fl, int *data ) {
  FILE *fptr;
  fptr = fopen ( fl, "r" );
  int value;
  int i = 0;
  while ( fscanf ( fptr, "%i", &value ) == 1 ) {
    data[i] = value;
    i += 1;
  }
  fclose ( fptr );
}

__host__ void SimpleWriteDataFloat ( const char *fl, const int n, const float *x ) {
  FILE *fptr;
  fptr = fopen ( fl, "w" );
  for ( int i = 0; i < n; i++ ) {
    fprintf ( fptr, " %.8E\n", x[i] );
  }
  fclose ( fptr );
}

__host__ void SimpleWriteDataFloat2D ( const char *fl, const int ns, const int nw, const float *x ) {
  FILE *fptr;
  fptr = fopen ( fl, "w" );
  for ( int j = 0; j < ns; j++ ) {
    for ( int i = 0; i < nw; i++ ) {
      fprintf ( fptr, " %.8E ", x[i+j*nw] );
    }
    fprintf ( fptr,  "\n" );
  }
  fclose ( fptr );
}

__host__ void AutocorrelationFunctionAveraged ( cufftResult_t cufftRes, cublasStatus_t cublasStat, cublasHandle_t cublasHandle, cufftHandle cufftPlan, const int nmbrOfStps, const int nmbrOfWlkrs, const float *chnFnctn, float *atcrrFnctn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int thrdsPerBlck = THRDSPERBLCK;
  dim3 dimBlock ( thrdsPerBlck, thrdsPerBlck );
  int blcksPerThrd_0 = ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck;
  int blcksPerThrd_1 = ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck;
  dim3 dimGrid_0 ( ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck );
  dim3 dimGrid_1 ( ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
  dim3 dimGrid_2 ( ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck );
  float *stps, *smOfChn, *cntrlChnFnctn, *wlkrs, *cmSmMtrx;
  cufftComplex *ftOfChn;
  cudaMallocManaged ( ( void ** ) &stps, nmbrOfStps * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &smOfChn, nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &cntrlChnFnctn, nmbrOfStps * nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &wlkrs, nmbrOfWlkrs * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &ftOfChn, nmbrOfStps * nmbrOfWlkrs * sizeof ( cufftComplex ) );
  cudaMallocManaged ( ( void ** ) &cmSmMtrx, nmbrOfStps * nmbrOfStps * sizeof ( float ) );
  ReturnConstantArray <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfStps, alpha / nmbrOfStps, stps );
  cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_N, nmbrOfWlkrs, nmbrOfStps, &alpha, chnFnctn, nmbrOfWlkrs, stps, incxx, &beta, smOfChn, incyy );
  if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: " ); }
  ReturnCentralChainFunction <<< dimGrid_0, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, smOfChn, chnFnctn, cntrlChnFnctn );
  ReturnChainFunctionTest <<< dimGrid_0, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 0, cntrlChnFnctn, ftOfChn );
  cufftRes = cufftExecC2C ( cufftPlan, ( cufftComplex * ) ftOfChn, ( cufftComplex * ) ftOfChn, CUFFT_FORWARD );
  if ( cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error:" ); }
  ComplexPointwiseMultiplyByConjugateAndScale <<< dimGrid_1, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, alpha / nmbrOfStps, ftOfChn );
  cufftRes = cufftExecC2C ( cufftPlan, ( cufftComplex * ) ftOfChn, ( cufftComplex * ) ftOfChn, CUFFT_INVERSE );
  if ( cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error: " ); }
  ReturnChainFunctionTest <<< dimGrid_0, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 1, cntrlChnFnctn, ftOfChn );
  ReturnConstantArray <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfWlkrs, alpha / nmbrOfWlkrs, wlkrs );
  cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, nmbrOfWlkrs, nmbrOfStps, &alpha, cntrlChnFnctn, nmbrOfWlkrs, wlkrs, incxx, &beta, atcrrFnctn, incyy );
  if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: " ); }
  NormalizeChain <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfStps, atcrrFnctn );
  //MakeMatrix <<< dimGrid_2, dimBlock >>> ( nmbrOfStps, atcrrFnctn, cmSmMtrx );
  //ReturnConstantArray <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfStps, alpha, stps );
  //cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, nmbrOfStps, nmbrOfStps, &alpha, cmSmMtrx, nmbrOfStps, stps, incxx, &beta, cmSmAtcrrFnctn, incyy );
  //if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: " ); }
  cudaFree ( stps );
  cudaFree ( smOfChn );
  cudaFree ( cntrlChnFnctn );
  cudaFree ( wlkrs );
  cudaFree ( ftOfChn );
  cudaFree ( cmSmMtrx );
}

__host__ void CumulativeSumOfAutocorrelationFunction ( const int ns, const float *x, float *c ) {
  float sum = 0;
  for ( int i = 0; i < ns; i++ ) {
    sum += x[i];
    c[i] = sum;
  }
}

__host__ int ChooseWindow ( const int n, const float c, const float *cm ) {
  int m = 0;
  while ( m < c * ( 2 * cm[m] - 1e0f ) && m < n ) {
    m += 1;
  }
  return m;
}

/* Kernels: */
__global__ void AssembleArrayOfRandomWalkers ( const int n, const float *rv, Walker *rw ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    for ( int p = 0; p < NPRS; p++ ) {
      rw[i].par[p] = rv[p+i*NPRS];
    }
  }
}

__global__ void AssembleArrayOfRandom2DWalkersFromTwoRandomArrays ( const int n, const float *a, const float *b, Walker *w ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    w[i].par[0] = a[i];
    w[i].par[1] = b[i];
  }
}

__global__ void InitializeWalkersAtRandom ( const int n, const float d, Walker sw, Walker *rw, Walker *wlk, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    wlk[i] = AddWalkers ( sw, ScaleWalker ( rw[i], d ) );
    stt[i] = 0;
  }
}

__global__ void InitializeWalkersAndStatisticsFromLastChain ( const int n, const float *chn, Walker *wlk, float *stt, float *prr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int p;
  if ( i < n ) {
    p = 0;
    while ( p < NPRS ) {
      wlk[i].par[p] = chn[p+i*(NPRS+2)];
      p += 1;
    }
    stt[i] = chn[p+i*(NPRS+2)];
    p += 1;
    prr[i] = chn[p+i*(NPRS+2)];
  }
}

__global__ void WriteWalkersAndStatisticsToChain ( const int n, const int ist, const Walker *wlk, const float *stt, const float *prr, Walker *wchn, float *schn, float *pchn ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    wchn[i+ist*n] = wlk[i];
    schn[i+ist*n] = stt[i];
    pchn[i+ist*n] = prr[i];
  }
}

__global__ void GenerateProposal ( const int n, const int ist, const int isb, const Walker *wlk, const float *rnd, float *zrnd, Walker *pwlk, float *pstt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int ts = i + isb * n;
  int r, tr, tc, k;
  float zz;
  Walker B;
  if ( i < n ) {
    r = 0;
    tr = i + r * n + isb * 3 * n + ist * 3 * 2 * n;
    zz = 1. / ACONST * powf ( rnd[tr] * ( ACONST - 1 ) + 1, 2. );
    zrnd[i] = zz;
    r = 1;
    tr = i + r * n + isb * 3 * n + ist * 3 * 2 * n;
    k = ( int ) truncf ( rnd[tr] * ( n - 1 + 0.999999 ) );
    tc = k + ( 1 - isb ) * n;
    B = AddWalkers ( wlk[ts], ScaleWalker ( wlk[tc], -1. ) );
    pwlk[i] = AddWalkers ( wlk[tc], ScaleWalker ( B, zz ) );
    pstt[i] = 0;
  }
}

__global__ void GenerateMetropolis ( const int n, const int ist, const int ipr, const Walker *wlk, const Walker *rwlk, Walker *pwlk, float *pstt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    pwlk[i] = wlk[i];
    pwlk[i].par[ipr] = wlk[i].par[ipr] + rwlk[i+n*ist].par[ipr];
    pstt[i] = 0;
  }
}

__global__ void UpdateWalkers ( const int nmbrOfHlfTheWlkrs, const int stpIndx, const int sbstIndx, const Walker *pwlk, const float *pstt, const float *pprr, const float *zRndmVls, const float *rndmVls, Walker *wlk, float *stt, float *prr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int t = i + sbstIndx * nmbrOfHlfTheWlkrs;
  int rnIndx = 2;
  int r = i + rnIndx * nmbrOfHlfTheWlkrs + sbstIndx * 3 * nmbrOfHlfTheWlkrs + stpIndx * 3 * nmbrOfHlfTheWlkrs;
  float q;
  if ( i < nmbrOfHlfTheWlkrs ) {
    q = - 0.5 * ( pstt[i] + pprr[i] - stt[t] - prr[t] );
    q = expf ( q ) * powf ( zRndmVls[i], NPRS - 1 );
    if ( q > rndmVls[r] ) {
      wlk[t] = pwlk[i];
      stt[t] = pstt[i];
      prr[t] = pprr[i];
    }
  }
}

__global__ void MetropolisUpdateOfWalkers ( const int n, const int stpIndx, const Walker *pwlk, const float *pstt, const float *pprr, const float *rndmVls, Walker *wlk, float *stt, float *prr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float q;
  if ( i < n ) {
    q = - 0.5 * ( pstt[i] + pprr[i] - stt[i] - prr[i] );
    q = expf ( q );
    if ( q > rndmVls[i+n*stpIndx] ) {
      wlk[i] = pwlk[i];
      stt[i] = pstt[i];
      prr[i] = pprr[i];
    }
  }
}

__global__ void ComplexPointwiseMultiplyByConjugateAndScale ( const int ns, const int nw, const float scl, Complex *a ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * ns;
  if ( j < nw && i < ns ) {
    a[t] = ScaleComplex ( MultiplyComplex ( a[t], ConjugateComplex ( a[t] ) ), scl );
  }
}

__global__ void ReturnChainFunctionTest ( const int ns, const int nw, const int sw, float *chn, Complex *a  ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int ttIndx0 = i + j * nw;
  int ttIndx1 = j + i * ns;
  if ( i < nw && j < ns ) {
    if ( sw == 0 ) {
      a[ttIndx1].x = chn[ttIndx0];
      a[ttIndx1].y = 0;
    } else if ( sw == 1 ) {
      chn[ttIndx0] = a[ttIndx1].x;
    }
  }
}

__global__ void ReturnChainFunction ( const int ns, const int nw, const int ipr, const Walker *cw, float *cf ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if ( i < nw && j < ns ) {
    cf[i+j*nw] = cw[i+j*nw].par[ipr];
  }
}

__global__ void ReturnConstantArray ( const int n, const float c, float *a ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    a[i] = c;
  }
}

__global__ void ReturnCentralChainFunction ( const int ns, const int nw, const float *scf, const float *cf, float *ccf ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if ( i < nw && j < ns ) {
    ccf[i+j*nw] = cf[i+j*nw] - scf[i];
  }
}

__global__ void NormalizeChain ( const int n, float *c ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    c[i] = c[i] / c[0];
  }
}

__global__ void MakeMatrix ( const int n, const float *chn, float *m ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.x;
  if ( i < n && j < n ) {
    m[i+j*n] = ( i <= j ) * chn[i];
  }
}

__global__ void BilinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int tIndx, const int grIndx, const float *data, const float *xin, const float *yin, const int M1, const int M2, const float *enrgChnnls, const Walker *wlkrs, float *mdlFlxs ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  float xxout, yyout, a, b, d00, d01, d10, d11, tmp1, tmp2, tmp3;
  int v, w;
  if ( i < nmbrOfEnrgChnnls && j < nmbrOfWlkrs ) {
    xxout = log10f ( enrgChnnls[i] );
    yyout = wlkrs[j].par[tIndx];
    v = FindElementIndex ( xin, M1, xxout );
    w = FindElementIndex ( yin, M2, yyout );
    a = ( xxout - xin[v] ) / ( xin[v+1] - xin[v] );
    b = ( yyout - yin[w] ) / ( yin[w+1] - yin[w] );
    if ( v < M1 && w < M2 ) d00 = data[w*M1+v]; else d00 = 0.;
    if ( v+1 < M1 && w < M2 ) d10 = data[w*M1+v+1]; else d10 = 0;
    if ( v < M1 && w+1 < M2 ) d01 = data[(w+1)*M1+v]; else d01 = 0;
    if ( v+1 < M1 && w+1 < M2 ) d11 = data[(w+1)*M1+v+1]; else d11 = 0;
    tmp1 = a * d10 + ( -d00 * a + d00 );
    tmp2 = a * d11 + ( -d01 * a + d01 );
    tmp3 = b * tmp2 + ( -tmp1 * b + tmp1 );
    mdlFlxs[i+j*nmbrOfEnrgChnnls] = powf ( 10., tmp3 );
  }
}

__global__ void LinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int dIndx, const float *Dist, const float *EBV, const float *errEBV, const Walker *wlkrs, float *mNh, float *sNh ) {
  int w = threadIdx.x + blockDim.x * blockIdx.x;
  float xxout, a, dmNh0, dmNh1, dsNh0, dsNh1, tmpMNh, tmpSNh;
  int v;
  if ( w < nmbrOfWlkrs ) {
    xxout = wlkrs[w].par[dIndx];
    v = FindElementIndex ( Dist, nmbrOfDistBins, xxout );
    a = ( xxout - Dist[v] ) / ( Dist[v+1] - Dist[v] );
    if ( v < nmbrOfDistBins ) dmNh0 = EBV[v]; else dmNh0 = 0;
    if ( v+1 < nmbrOfDistBins ) dmNh1 = EBV[v+1]; else dmNh1 = 0;
    tmpMNh = a * dmNh1 + ( -dmNh0 * a + dmNh0 );
    if ( v < nmbrOfDistBins ) dsNh0 = errEBV[v]; else dsNh0 = 0;
    if ( v+1 < nmbrOfDistBins ) dsNh1 = errEBV[v+1]; else dsNh1 = 0;
    tmpSNh = a * dsNh1 + ( -dsNh0 * a + dsNh0 );
    tmpMNh = powf ( 10, tmpMNh );
    tmpSNh = powf ( 10, tmpSNh );
    mNh[w] = 0.8 * tmpMNh;
    sNh[w] = 0.8 * tmpMNh * ( powf ( tmpSNh / tmpMNh, 2 ) + powf ( 0.3 / 0.8, 2 ) );
  }
}

__host__ int printMove ( const int ist, const int isb, const Chain *chn ) {
  printf ( "=========================================\n" );
  printf ( " step - %i ", ist );
  printf ( " subset - %i: ", isb );
  printf ( "\n" );
  printf ( "=========================================\n" );
  printf ( " random -- ")
  printf ( "\n" );
  int rr = isb * 3 * chn[0].nmbrOfWlkrs/2 + ist * 3 * 2 * chn[0].nmbrOfWlkrs/2;
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    r = i + 0 * chn[0].nmbrOfWlkrs/2 + rr;
    printf ( " %2.4f ", chn[0].rndmVls[r] );
  }
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    r = i + 1 * chn[0].nmbrOfWlkrs/2 + rr;
    printf ( " %2.4f ", chn[0].rndmVls[r] );
  }
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    r = i + 2 * chn[0].nmbrOfWlkrs/2 + rr;
    printf ( " %2.4f ", chn[0].rndmVls[r] );
  }
  printf ( "\n" );
  printf ( " xx -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NPRS; i++ ) {
    for ( int j = 0; j < chn[0].nmbrOfWlkrs; j++ ) {
      printf ( " %2.4f ", chn[0].wlkrs[j].par[i] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs; i++ ) {
    printf ( " %2.4f ", chn[0].sttstcs[i] );
  }
  printf ( "\n" );
  printf ( " xx0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NPRS; i++ ) {
    for ( int j = isb*chn[0].nmbrOfWlkrs/2; j < (1+isb)*chn[0].nmbrOfWlkrs/2; j++ ) {
      printf ( " %2.4f ", chn[0].wlkrs[j].par[i] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt0 -- "  );
  printf ( "\n" );
  for ( int i = isb*chn[0].nmbrOfWlkrs/2; i < (1+isb)*chn[0].nmbrOfWlkrs/2; i++ ) {
    printf ( " %2.4f ", chn[0].sttstcs[i] );
  }
  printf ( "\n" );
  printf ( " xxC -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NPRS; i++ ) {
    for ( int j = (1-isb)*chn[0].nmbrOfWlkrs/2; j < (2-isb)*chn[0].nmbrOfWlkrs/2; j++ ) {
      printf ( " %2.4f ", chn[0].wlkrs[j].par[i] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );

  printf ( " kr -- "  );
  printf ( "\n" );
  int r;
  int k;
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    r = i + 1 * chn[0].nmbrOfWlkrs/2 + isb * 3 * chn[0].nmbrOfWlkrs/2 + ist * 3 * 2 * chn[0].nmbrOfWlkrs/2 ;
    k = ( int ) truncf ( chn[0].rndmVls[r] * ( chn[0].nmbrOfWlkrs/2 - 1 + 0.999999 ) );
    printf ( " %i ", k );
  }
  printf ( "\n" );
  printf ( " xxCP -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NPRS; i++ ) {
    for ( int j = 0; j < chn[0].nmbrOfWlkrs/2; j++ ) {
      r = i + 1 * chn[0].nmbrOfWlkrs/2 + isb * 3 * chn[0].nmbrOfWlkrs/2 + ist * 3 * 2 * chn[0].nmbrOfWlkrs/2;
      k = ( int ) truncf ( chn[0].rndmVls[r] * ( chn[0].nmbrOfWlkrs/2 - 1 + 0.999999 ) );
      printf ( " %2.4f ", chn[0].wlkrs[k+(1-isb)*chn[0].nmbrOfWlkrs/2].par[i] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );

  printf ( " zr -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    printf ( " %2.4f ", chn[0].zRndmVls[i] );
  }
  printf ( "\n" );
  printf ( " xx1 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NPRS; i++ ) {
    for ( int j = 0; j < chn[0].nmbrOfWlkrs/2; j++ ) {
      printf ( " %2.4f ", chn[0].prpsdWlkrs[j].par[i] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  return 0;
}

__host__ int printUpdate ( const int ist, const int isb, const Chain *chn ) {
  printf ( "------------------------------------------\n" );
  printf ( " stt1 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    printf ( " %2.4f ", chn[0].prpsdSttstcs[i] );
  }
  printf ( "\n" );
  printf ( " q -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    printf ( " %2.4f ", expf ( -0.5 * ( chn[0].prpsdSttstcs[i] - chn[0].sttstcs[i+isb*chn[0].nmbrOfWlkrs] ) ) );
  }
  printf ( "\n" );
  printf ( " ru -- "  );
  printf ( "\n" );
  int r;
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    r = i + 2 * chn[0].nmbrOfWlkrs/2 + isb * 3 * chn[0].nmbrOfWlkrs/2 + ist * 3 * 2 * chn[0].nmbrOfWlkrs/2 ;
    printf ( " %2.4f ", chn[0].rndmVls[r] );
  }
  printf ( "\n" );
  printf ( " xx0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NPRS; i++ ) {
    for ( int j = 0; j < chn[0].nmbrOfWlkrs/2; j++ ) {
      printf ( " %2.4f ", chn[0].wlkrs[j].par[i] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nmbrOfWlkrs/2; i++ ) {
    printf ( " %2.4f ", chn[0].sttstcs[i] );
  }
  printf ( "\n" );
  return 0;
}

#endif // _STRCTRSANDFNCTNS_CU_
