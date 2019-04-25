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

__host__ int grid1D ( const int n ) {
  int b = ( n + THRDSPERBLCK - 1 ) / THRDSPERBLCK;
  return b;
}

__host__ dim3 grid2D ( const int n, const int m ) {
  dim3 grid ( ( n + THRDSPERBLCK - 1 ) / THRDSPERBLCK, ( m + THRDSPERBLCK - 1 ) / THRDSPERBLCK );
  return grid;
}

__host__ dim3 block2D () {
  dim3 block ( THRDSPERBLCK, THRDSPERBLCK );
  return block;
}

__host__ __device__ Complex addComplex ( Complex a, Complex b ) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

__host__ __device__ Complex scaleComplex ( Complex a, float s ) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

__host__ __device__ Complex multiplyComplex ( Complex a, Complex b ) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__host__ __device__ Complex conjugateComplex ( Complex a ) {
  Complex c;
  c.x = a.x;
  c.y = - a.y;
  return c;
}

__global__ void constantArray ( const int n, const float c, float *a ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    a[i] = c;
  }
}

__global__ void sliceArray ( const int n, const int indx, const float *ss, float *zz ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    zz[i] = ss[i+indx];
  }
}

__global__ void sliceIntArray ( const int n, const int indx, const int *ss, int *zz ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    zz[i] = ss[i+indx];
  }
}

__global__ void insertArray ( const int n, const int indx, const float *ss, float *zz ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    zz[indx+i] = ss[i];
  }
}

__global__ void initializeAtRandom ( const int dim, const int nwl, const float dlt, const float *x0, const float *stn, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx[t] = x0[i] + dlt * stn[t];
  }
}

__global__ void returnStatistic ( const int dim, const int nwl, const float *xx, float *s ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    s[t] = powf ( xx[t], 2. );
  }
}

__global__ void setWalkersAtLast ( const int dim, const int nwl, const float *lst, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx[t] = lst[i+j*(dim+1+1+1+1)];
  }
}

__global__ void setStatisticAtLast ( const int dim, const int nwl, const float *lst, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stt[i] = lst[dim+1+i*(dim+1+1+1+1)];
  }
}

__global__ void setChiAtLast ( const int dim, const int nwl, const float *lst, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stt[i] = lst[dim+2+i*(dim+1+1+1+1)];
  }
}

__global__ void setDistanceAtLast ( const int dim, const int nwl, const float *lst, float *didi ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    didi[i] = lst[dim+i*(dim+1+1+1+1)];
  }
}

__global__ void setPriorAtLast ( const int dim, const int nwl, const float *lst, float *prr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    prr[i] = lst[dim+3+i*(dim+1+1+1+1)];
  }
}

__global__ void complexPointwiseMultiplyByConjugateAndScale ( const int nst, const int nwl, const float scl, Complex *a ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nst;
  if ( i < nst && j < nwl ) {
    a[t] = scaleComplex ( multiplyComplex ( a[t], conjugateComplex ( a[t] ) ), scl );
  }
}

__global__ void testChainFunction ( const int nwl, const int nst, const int sw, float *chn, Complex *a  ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t0 = i + j * nwl;
  int t1 = j + i * nst;
  if ( i < nwl && j < nst ) {
    if ( sw == 0 ) { a[t1].x = chn[t0]; a[t1].y = 0; }
    else if ( sw == 1 ) { chn[t0] = a[t1].x; }
  }
}

__global__ void chainFunction ( const int dim, const int nwl, const int nst, const int ipr, const float *smpls, float *chnFnctn ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nwl;
  if ( i < nwl && j < nst ) {
    chnFnctn[t] = smpls[ipr+t*dim];
  }
}

__global__ void normArray ( const int n, float *a ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float c = a[0];
  if ( i < n ) {
    a[i] = a[i] / c;
  }
}

__global__ void scaleArray ( const int n, const float c, float *a ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    a[i] = c * a[i];
  }
}

__global__ void shiftWalkers ( const int dim, const int nwl, const float *xx, const float *x, float *yy ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    yy[t] = xx[t] - x[i];
  }
}

__global__ void addWalkers ( const int dim, const int nwl, const float *xx0, const float *xxW, float *xx1 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx1[t] = xx0[t] + xxW[t];
  }
}

__global__ void returnQ ( const int dim, const int n, const float *s1, const float *s0, const float *zr, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    q[i] = expf ( - 0.5 * ( s1[i] - s0[i] ) ) * powf ( zr[i], dim - 1 );
  }
}

__global__ void returnQ1 ( const int dim, const int n, const float *p1, const float *p0, const float *s1, const float *s0, const float *zr, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    if ( p1[i] == INF || - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) < -10. ) {
      q[i] = 0.0;
    } else if ( - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) > 10. ) {
      q[i] = 1.E10;
    } else {
      q[i] = expf ( - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) ) * powf ( zr[i], dim - 1 );
    }
  }
}

__global__ void returnQM ( const int dim, const int n, const float *s1, const float *s0, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    q[i] = expf ( - 0.5 * ( s1[i] - s0[i] ) );
  }
}

__global__ void updateWalkers ( const int dim, const int nwl, const float *xx1, const float *q, const float *r, float *xx0 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    //if ( q[j] > r[j] ) {
    xx0[t] = ( q[j] > r[j] ) * xx1[t] + ( q[j] <= r[j] ) * xx0[t];
    //}
  }
}

__global__ void updateStatistic ( const int nwl, const float *stt1, const float *q, const float *r, float *stt0 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stt0[i] = ( q[i] > r[i] ) * stt1[i] + ( q[i] < r[i] ) * stt0[i];
  }
}

__global__ void saveWalkers ( const int dim, const int nwl, const int ist, const float *xx, float *smpls ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    smpls[t+ist*dim*nwl] = xx[t];
  }
}

__global__ void saveStatistic ( const int nwl, const int ist, const float *stt, float *stat ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stat[i+ist*nwl] = stt[i];
  }
}

__global__ void mapRandomNumbers ( const int nwl, const int ist, const int isb, const float *r, float *zr, int *kr, float *ru, int *kex ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int rr;
  if ( i < nwl ) {
    rr = i + 0 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
    zr[i] = 1. / ACONST * powf ( r[rr] * ( ACONST - 1 ) + 1, 2. );
    rr = i + 1 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
    kr[i] = ( int ) truncf ( r[rr] * ( nwl - 1 + 0.999999 ) );
    rr = i + 2 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
    ru[i] = r[rr];
    rr = i + 3 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
    kex[i] = ( int ) truncf ( r[rr] * ( 3 - 1 + 0.999999 ) );
  }
}

__global__ void mapKex ( const int nwl, const float *r, int *kex ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    kex[i] = ( int ) truncf ( r[i] * ( 3 - 1 + 0.999999 ) );
  }
}


__global__ void TestpermuteWalkers ( const int dim, const int nwl, const int *kr, const float *xxC, float *xxCP ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xxCP[t] = xxC[t];
  }
}


__global__ void permuteWalkers ( const int dim, const int nwl, const int *kr, const float *xxC, float *xxCP ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  int p = i + kr[j] * dim;
  if ( i < dim && j < nwl ) {
    xxCP[t] = xxC[p];
  }
}

__global__ void substractWalkers ( const int dim, const int nwl, const float *xx0, const float *xxCP, float *xx1 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx1[t] = xx0[t] - xxCP[t];
  }
}

__global__ void scale2DArray ( const int dim, const int nwl, const float *zr, const float *xx, float *xx1 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx1[t] = zr[j] * xx[t];
  }
}

__global__ void metropolisPoposal2 ( const int dim, const int nwl, const int isb, const float *xx, const float *rr, float *xx1 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx1[t] = xx[t] + ( i == isb ) * rr[j];
  }
}

__global__ void chooseLaw ( const int nwl, const int *kex, const float *didi11, const float *didi12, const float *didi13, float *didi1 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    didi1[i] = ( kex[i] == 0 ) * didi11[i] + ( kex[i] == 1 ) * didi12[i] + ( kex[i] == 2 ) * didi13[i];
  }
}

__host__ int initializeCuda ( Cupar *cdp ) {
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
  return 0;
}

__host__ int allocateChain ( Chain *chn ) {
  cudaMallocManaged ( ( void ** ) &chn[0].stn, chn[0].nst * 2 * chn[0].nwl * chn[0].dim * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].stn1, chn[0].nst * chn[0].nwl * chn[0].dim * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].uni, chn[0].dim * chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].lst, ( chn[0].dim + 1 ) * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].x0, chn[0].dim * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].zz, chn[0].nwl * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ru, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].rr, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].stt, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].stt1, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].sstt1, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].sstt, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].stt0, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].q, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xx, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xx0, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xxC, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xx1, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xxCM, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xCM, chn[0].dim * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xxW, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].wcnst, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].dcnst, chn[0].dim * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].smpls, chn[0].dim * chn[0].nwl * chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].stat, chn[0].nwl * chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].priors, chn[0].nwl * chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xxCP, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].zr, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].zuni, chn[0].nst * 2 * chn[0].nwl / 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].kr, chn[0].nwl * sizeof ( int ) );
  cudaMallocManaged ( ( void ** ) &chn[0].kex, chn[0].nwl * sizeof ( int ) );
  cudaMallocManaged ( ( void ** ) &chn[0].kuni, chn[0].nst * 2 * chn[0].nwl / 2 * sizeof ( int ) );
  cudaMallocManaged ( ( void ** ) &chn[0].runi, chn[0].nst * 2 * chn[0].nwl / 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].stps, chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].smOfChn, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cntrlChnFnctn, chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chnFnctn, chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ftOfChn, chn[0].nst * chn[0].nwl * sizeof ( cufftComplex ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cmSmMtrx, chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].atcrrFnctn, chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cmSmAtCrrFnctn, chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prr, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prr1, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prr0, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xbnd, chn[0].dim * 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cnd, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ccnd, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].nhMd, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].nhSg, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi11, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi12, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi13, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi01, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi02, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi03, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi0, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].dist, chn[0].nwl * chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].didi1, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chi1, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chi, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chi0, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chiTwo, chn[0].nwl * chn[0].nst * sizeof ( float ) );
  return 0;
}

__host__ int initializeChain ( Cupar *cdp, Chain *chn, Model *mdl, Spectrum *spc ) {
  constantArray <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, 1., chn[0].wcnst );
  constantArray <<< grid1D ( chn[0].dim ), THRDS >>> ( chn[0].dim, 1., chn[0].dcnst );
  if ( chn[0].indx == 0 ) {
    curandGenerateNormal ( cdp[0].curandGnrtr, chn[0].stn, chn[0].dim * chn[0].nwl, 0, 1 );
    initializeAtRandom <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].dlt, chn[0].x0, chn[0].stn, chn[0].xx );
    constantArray <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, 0., chn[0].stt );
    constantArray <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, 0., chn[0].chi );
    curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].uni, chn[0].nwl );
    mapKex <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, chn[0].uni, chn[0].kex );
    //statistic0 ( cdp, chn );
    modelStatistic0 ( cdp, mdl, chn, spc );
  } else {
    readLastFromFile ( chn[0].name, chn[0].indx-1, chn[0].dim, chn[0].nwl, chn[0].lst );
    setWalkersAtLast <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].xx );
    setStatisticAtLast <<< grid1D ( chn[0].nwl ), THRDS  >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].stt );
    setChiAtLast <<< grid1D ( chn[0].nwl ), THRDS  >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].chi );
    setPriorAtLast <<< grid1D ( chn[0].nwl ), THRDS  >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].prr );
    setDistanceAtLast <<< grid1D ( chn[0].nwl ), THRDS  >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].didi );
  }
  return 0;
}

__host__ int initializeRandomForWalk ( Cupar *cdp, Chain *chn ) {
  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].uni, chn[0].nst * 2 * chn[0].nwl / 2 );
  curandGenerateNormal ( cdp[0].curandGnrtr, chn[0].stn, chn[0].nst * 2 * chn[0].nwl / 2 * chn[0].dim, 0, 1 );
  return 0;
}

__host__ int initializeRandomForStreach ( Cupar *cdp, Chain *chn ) {
  int n = chn[0].nst * 2 * 4 * chn[0].nwl / 2;
  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].uni, n );
  return 0;
}

__host__ int initializeRandomForMetropolis ( Cupar *cdp, Chain *chn ) {
  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].uni, chn[0].nst * chn[0].nwl * chn[0].dim );
  curandGenerateNormal ( cdp[0].curandGnrtr, chn[0].stn1, chn[0].nst * chn[0].nwl * chn[0].dim, 0, 1 );
  return 0;
}

__host__ int walkMove ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int indxXC = ( 1 - chn[0].isb ) * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  int nrn = chn[0].nwl / 2 * chn[0].nwl / 2;
  int indxRn = chn[0].ist * 2 * nrn + chn[0].isb * nrn;
  int nru = chn[0].nwl / 2;
  int indxRu = chn[0].ist * 2 * nru + chn[0].isb * nru;
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].xx, chn[0].xx0 );
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxXC, chn[0].xx, chn[0].xxC );
  sliceArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt, chn[0].stt0 );
  sliceArray <<< grid1D ( nrn ), THRDSPERBLCK >>> ( nrn, indxRn, chn[0].stn, chn[0].zz );
  sliceArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].uni, chn[0].ru );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_N, chn[0].dim, chn[0].nwl/2, &alpha, chn[0].xxC, chn[0].dim, chn[0].wcnst, incxx, &beta, chn[0].xCM, incyy );
  scaleArray <<< grid1D ( chn[0].dim ), THRDSPERBLCK >>> ( chn[0].dim, 2./chn[0].nwl, chn[0].xCM );
  shiftWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xxC, chn[0].xCM, chn[0].xxCM );
  cublasSgemm ( cdp[0].cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, chn[0].dim, chn[0].nwl/2 , chn[0].nwl/2, &alpha, chn[0].xxCM, chn[0].dim, chn[0].zz, chn[0].nwl/2, &beta, chn[0].xxW, chn[0].dim );
  addWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx0, chn[0].xxW, chn[0].xx1 );
  return 0;
}

__host__ int streachMove ( const Cupar *cdp, Chain *chn ) {
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int indxXC = ( 1 - chn[0].isb ) * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  //int nru = chn[0].nwl / 2;
  //int indxRu = chn[0].isb * chn[0].nwl/2 + chn[0].ist * 2 * chn[0].nwl/2;
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].xx, chn[0].xx0 );
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxXC, chn[0].xx, chn[0].xxC );
  sliceArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt, chn[0].stt0 );
  sliceArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].prr, chn[0].prr0 );
  sliceArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].chi, chn[0].chi0 );
  sliceArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].didi, chn[0].didi0 );
  //sliceArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].zuni, chn[0].zr );
  //sliceIntArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].kuni, chn[0].kr );
  mapRandomNumbers <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].ist, chn[0].isb, chn[0].uni, chn[0].zr, chn[0].kr, chn[0].ru, chn[0].kex );
  //sliceArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].runi, chn[0].ru );
  permuteWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].kr, chn[0].xxC, chn[0].xxCP );
  substractWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx0, chn[0].xxCP, chn[0].xxCM );
  scale2DArray <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].zr, chn[0].xxCM, chn[0].xxW );
  addWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xxCP, chn[0].xxW, chn[0].xx1 );
  return 0;
}

__host__ int metropolisMove ( const Cupar *cdp, Chain *chn ) {
  int nrn = chn[0].nwl;
  int iRn = chn[0].isb * chn[0].nwl + chn[0].ist * chn[0].dim * chn[0].nwl;
  int nru = chn[0].nwl;
  int iRu = chn[0].isb * chn[0].nwl + chn[0].ist * chn[0].dim * chn[0].nwl;
  sliceArray <<< grid1D ( nrn ), THRDSPERBLCK >>> ( nrn, iRn, chn[0].stn1, chn[0].rr );
  sliceArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, iRu, chn[0].uni, chn[0].ru );
  metropolisPoposal2 <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].isb, chn[0].xx, chn[0].rr, chn[0].xx1 );
  return 0;
}

__host__ int statistic ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  returnStatistic <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx1, chn[0].sstt1 );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl/2, &alpha, chn[0].sstt1, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].stt1, incyy );
  return 0;
}

__host__ int statisticMetropolis ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  returnStatistic <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xx1, chn[0].sstt1 );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl, &alpha, chn[0].sstt1, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].stt1, incyy );
  return 0;
}

__host__ int statistic0 ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  returnStatistic <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xx, chn[0].sstt );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl, &alpha, chn[0].sstt, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].stt, incyy );
  return 0;
}

__host__ int walkUpdate ( const Cupar *cdp, Chain *chn ) {
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  returnQM <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].dim, chn[0].nwl/2, chn[0].stt1, chn[0].stt0, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].xx0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt0 );
  insertArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].xx0, chn[0].xx );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt0, chn[0].stt );
  return 0;
}

__host__ int metropolisUpdate ( const Cupar *cdp, Chain *chn ) {
  returnQM <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].dim, chn[0].nwl, chn[0].stt1, chn[0].stt0, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].xx );
  updateStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt );
  return 0;
}

__host__ int streachUpdate ( const Cupar *cdp, Chain *chn, Model *mdl ) {
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  returnQ1 <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].dim, chn[0].nwl/2, chn[0].prr1, chn[0].prr0, chn[0].stt1, chn[0].stt0, chn[0].zr, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].xx0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].prr1, chn[0].q, chn[0].ru, chn[0].prr0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].chi1, chn[0].q, chn[0].ru, chn[0].chi0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].didi1, chn[0].q, chn[0].ru, chn[0].didi0 );
  insertArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].xx0, chn[0].xx );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt0, chn[0].stt );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].chi0, chn[0].chi );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].prr0, chn[0].prr );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].didi0, chn[0].didi );
  return 0;
}

__host__ int saveCurrent ( Chain *chn ) {
  saveWalkers <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].ist, chn[0].xx, chn[0].smpls );
  saveStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].ist, chn[0].stt, chn[0].stat );
  saveStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].ist, chn[0].chi, chn[0].chiTwo );
  saveStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].ist, chn[0].prr, chn[0].priors );
  saveStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].ist, chn[0].didi, chn[0].dist );
  return 0;
}

__host__ int averagedAutocorrelationFunction ( Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int NN[RANK] = { chn[0].nst };
  cufftPlanMany ( &cdp[0].cufftPlan, RANK, NN, NULL, 1, chn[0].nst, NULL, 1, chn[0].nst, CUFFT_C2C, chn[0].nwl );
  chainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].nst, 0, chn[0].smpls, chn[0].chnFnctn );
  constantArray <<< grid1D ( chn[0].nst ), THRDSPERBLCK >>> ( chn[0].nst, alpha / chn[0].nst, chn[0].stps );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_N, chn[0].nwl, chn[0].nst, &alpha, chn[0].chnFnctn, chn[0].nwl, chn[0].stps, incxx, &beta, chn[0].smOfChn, incyy );
  shiftWalkers <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].nwl, chn[0].nst, chn[0].chnFnctn, chn[0].smOfChn, chn[0].cntrlChnFnctn );
  testChainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].nwl, chn[0].nst, 0, chn[0].cntrlChnFnctn, chn[0].ftOfChn );
  cufftExecC2C ( cdp[0].cufftPlan, ( cufftComplex * ) chn[0].ftOfChn, ( cufftComplex * ) chn[0].ftOfChn, CUFFT_FORWARD );
  complexPointwiseMultiplyByConjugateAndScale <<< grid2D ( chn[0].nst, chn[0].nwl ), block2D () >>> ( chn[0].nst, chn[0].nwl, alpha / chn[0].nst, chn[0].ftOfChn );
  cufftExecC2C ( cdp[0].cufftPlan, ( cufftComplex * ) chn[0].ftOfChn, ( cufftComplex * ) chn[0].ftOfChn, CUFFT_INVERSE );
  testChainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].nwl, chn[0].nst, 1, chn[0].cntrlChnFnctn, chn[0].ftOfChn );
  constantArray <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, alpha / chn[0].nwl, chn[0].wcnst );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].nwl, chn[0].nst, &alpha, chn[0].cntrlChnFnctn, chn[0].nwl, chn[0].wcnst, incxx, &beta, chn[0].atcrrFnctn, incyy );
  //scaleArray <<< grid1D ( chn[0].nst ), THRDSPERBLCK >>> ( chn[0].nst, 1. / chn[0].atcrrFnctn[0], chn[0].atcrrFnctn );
  normArray <<< grid1D ( chn[0].nst ), THRDSPERBLCK >>> ( chn[0].nst, chn[0].atcrrFnctn );
  cudaDeviceSynchronize ();
  cumulativeSumOfAutocorrelationFunction ( chn[0].nst, chn[0].atcrrFnctn, chn[0].cmSmAtCrrFnctn );
  int MM = chooseWindow ( chn[0].nst, 5e0f, chn[0].cmSmAtCrrFnctn );
  chn[0].mmm = MM;
  chn[0].atcTime = 2 * chn[0].cmSmAtCrrFnctn[MM] - 1e0f;
  return 0;
}

__host__ void readLastFromFile ( const char *name, const int indx, const int dim, const int nwl, float *lst ) {
  FILE *fptr;
  char fl[FLEN_CARD];
  snprintf ( fl, sizeof ( fl ), "%s%i%s", name, indx, ".chain" );
  fptr = fopen ( fl, "r" );
  float value;
  int i = 0;
  while ( fscanf ( fptr, "%e", &value ) == 1 ) {
    i += 1;
  }
  fclose ( fptr );
  int n = i;
  fptr = fopen ( fl, "r" );
  i = 0;
  int j;
  while ( fscanf ( fptr, "%e", &value ) == 1 ) {
    if ( i >= n - ( dim + 1 + 1 + 1 + 1 ) * nwl ) {
      j = i - ( n - ( dim + 1 + 1 + 1 + 1  ) * nwl );
      lst[j] = value;
    }
    i += 1;
  }
  fclose ( fptr );
}

__host__ void writeChainToFile ( const char *name, const int indx, const int dim, const int nwl, const int nst, const float *smpls, const float *stat, const float *priors, const float *dist, const float *chi ) {
  FILE *flPntr;
  char flNm[FLEN_CARD];
  int ttlChnIndx, stpIndx, wlkrIndx, prmtrIndx;
  snprintf ( flNm, sizeof ( flNm ), "%s%i%s", name, indx, ".chain" );
  flPntr = fopen ( flNm, "w" );
  stpIndx = 0;
  while ( stpIndx < nst ) {
    wlkrIndx = 0;
    while ( wlkrIndx < nwl ) {
      ttlChnIndx = wlkrIndx * dim + stpIndx * nwl * dim;
      prmtrIndx = 0;
      while ( prmtrIndx < dim ) {
        fprintf ( flPntr, " %.8E ", smpls[prmtrIndx+ttlChnIndx] );
        prmtrIndx += 1;
      }
      fprintf ( flPntr, " %.8E ", dist[wlkrIndx+stpIndx*nwl] );
      prmtrIndx += 1;
      fprintf ( flPntr, " %.8E ", stat[wlkrIndx+stpIndx*nwl] );
      prmtrIndx += 1;
      fprintf ( flPntr, " %.8E ", chi[wlkrIndx+stpIndx*nwl] );
      prmtrIndx += 1;
      fprintf ( flPntr, " %.8E\n", priors[wlkrIndx+stpIndx*nwl] );
      wlkrIndx += 1;
    }
    stpIndx += 1;
  }
  fclose ( flPntr );
}

__host__ int destroyCuda ( const Cupar *cdp ) {
  cusparseDestroy ( cdp[0].cusparseHandle );
  cublasDestroy ( cdp[0].cublasHandle );
  curandDestroyGenerator ( cdp[0].curandGnrtr );
  curandDestroyGenerator ( cdp[0].curandGnrtrHst );
  cudaEventDestroy ( cdp[0].start );
  cudaEventDestroy ( cdp[0].stop );
  cufftDestroy ( cdp[0].cufftPlan );
  return 0;
}

__host__ void freeChain ( const Chain *chn ) {
  cudaFree ( chn[0].stn );
  cudaFree ( chn[0].uni );
  cudaFree ( chn[0].zz );
  cudaFree ( chn[0].stt );
  cudaFree ( chn[0].xx );
  cudaFree ( chn[0].xx0 );
  cudaFree ( chn[0].xxC );
  cudaFree ( chn[0].xx1 );
  cudaFree ( chn[0].xxCM );
  cudaFree ( chn[0].xCM );
  cudaFree ( chn[0].xxW );
  cudaFree ( chn[0].wcnst );
  cudaFree ( chn[0].dcnst );
  cudaFree ( chn[0].x0 );
  cudaFree ( chn[0].smpls );
  cudaFree ( chn[0].stat );
  cudaFree ( chn[0].stt1 );
  cudaFree ( chn[0].sstt1 );
  cudaFree ( chn[0].ru );
  cudaFree ( chn[0].q );
  cudaFree ( chn[0].stt0 );
  cudaFree ( chn[0].zr );
  cudaFree ( chn[0].kr );
  cudaFree ( chn[0].xxCP );
  cudaFree ( chn[0].zuni );
  cudaFree ( chn[0].kuni );
  cudaFree ( chn[0].runi );
  cudaFree ( chn[0].stps );
  cudaFree ( chn[0].smOfChn );
  cudaFree ( chn[0].cntrlChnFnctn );
  cudaFree ( chn[0].ftOfChn );
  cudaFree ( chn[0].cmSmMtrx );
  cudaFree ( chn[0].chnFnctn );
  cudaFree ( chn[0].atcrrFnctn );
  cudaFree ( chn[0].cmSmAtCrrFnctn );
  cudaFree ( chn[0].stn1 );
  cudaFree ( chn[0].rr );
  cudaFree ( chn[0].sstt );
  cudaFree ( chn[0].nhMd );
  cudaFree ( chn[0].nhSg );
  cudaFree ( chn[0].dist );
  cudaFree ( chn[0].didi );
  cudaFree ( chn[0].didi0 );
  cudaFree ( chn[0].didi1 );
  cudaFree ( chn[0].didi11 );
  cudaFree ( chn[0].didi12 );
  cudaFree ( chn[0].didi13 );
  cudaFree ( chn[0].didi01 );
  cudaFree ( chn[0].didi02 );
  cudaFree ( chn[0].didi03 );
  cudaFree ( chn[0].kex );
  cudaFree ( chn[0].chi1 );
  cudaFree ( chn[0].chi );
  cudaFree ( chn[0].chi0 );
  cudaFree ( chn[0].chiTwo );
}

__host__ void cumulativeSumOfAutocorrelationFunction ( const int nst, const float *chn, float *cmSmChn ) {
  float sum = 0;
  for ( int i = 0; i < nst; i++ ) {
    sum = sum + chn[i];
    cmSmChn[i] = sum;
  }
}

__host__ int chooseWindow ( const int nst, const float c, const float *cmSmChn ) {
  int m = 0;
  while ( m < c * ( 2 * cmSmChn[m] - 1e0f ) && m < nst  ) {
    m += 1;
  }
  return m-1;
}

__host__ void simpleReadDataFloat ( const char *fl, float *data ) {
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

__host__ void simpleReadDataInt ( const char *fl, int *data ) {
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

__host__ void simpleWriteDataFloat ( const char *fl, const int n, const float *x ) {
  FILE *fptr;
  fptr = fopen ( fl, "w" );
  for ( int i = 0; i < n; i++ ) {
    fprintf ( fptr, " %.8E\n", x[i] );
  }
  fclose ( fptr );
}

__host__ void simpleWriteDataFloat2D ( const char *fl, const int ns, const int nw, const float *x ) {
  FILE *fptr = fopen ( fl, "w" );
  for ( int j = 0; j < ns; j++ ) {
    for ( int i = 0; i < nw; i++ ) {
      fprintf ( fptr, " %.8E ", x[i+j*nw] );
    }
    fprintf ( fptr,  "\n" );
  }
  fclose ( fptr );
}

__host__ int printMetropolisMove ( const Chain *chn ) {
  printf ( "=========================================\n" );
  printf ( " step - %i ", chn[0].ist );
  printf ( " subset - %i: ", chn[0].isb );
  printf ( "\n" );
  printf ( "=========================================\n" );
  printf ( "\n" );
  printf ( " rr -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].rr[i] );
  }
  printf ( "\n" );
  printf ( " ru -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].ru[i] );
  }
  printf ( "\n" );
  printf ( " xx -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl; j++ ) {
      printf ( " %2.4f ", chn[0].xx[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].stt[i] );
  }
  printf ( "\n" );
  printf ( " xx1 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl; j++ ) {
      printf ( " %2.4f ", chn[0].xx1[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  return 0;
}

__host__ int printMetropolisUpdate ( const Chain *chn ) {
  printf ( "------------------------------------------\n" );
  printf ( " stt1 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].stt1[i] );
  }
  printf ( "\n" );
  printf ( " q -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].q[i] );
  }
  printf ( "\n" );
  return 0;
}

__host__ int printMove ( const Chain *chn ) {
  printf ( "=========================================\n" );
  printf ( " step - %i ", chn[0].ist );
  printf ( " subset - %i: ", chn[0].isb );
  printf ( "\n" );
  printf ( "=========================================\n" );
  printf ( " random -- ");
  printf ( "\n" );
  int rr = chn[0].isb * 3 * chn[0].nwl/2 + chn[0].ist * 3 * 2 * chn[0].nwl/2;
  int rrr;
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    rrr = i + 0 * chn[0].nwl/2 + rr;
    printf ( " %2.4f ", chn[0].uni[rrr] );
  }
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    rrr = i + 1 * chn[0].nwl/2 + rr;
    printf ( " %2.4f ", chn[0].uni[rrr] );
  }
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    rrr = i + 2 * chn[0].nwl/2 + rr;
    printf ( " %2.4f ", chn[0].uni[rrr] );
  }
  printf ( "\n" );
  printf ( " xx -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl; j++ ) {
      printf ( " %2.4f ", chn[0].xx[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].stt[i] );
  }
  printf ( "\n" );
  printf ( " xx0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].xx0[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %2.4f ", chn[0].stt0[i] );
  }
  printf ( "\n" );
  printf ( " xxC -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].xxC[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " kr -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %i ", chn[0].kr[i] );
  }
  printf ( "\n" );
  printf ( " xxCP -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].xxCP[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " zr -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %2.4f ", chn[0].zr[i] );
  }
  printf ( "\n" );
  printf ( " xx1 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].xx1[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  return 0;
}

__host__ int printUpdate ( const Chain *chn ) {
  printf ( "------------------------------------------\n" );
  printf ( "\n" );
  printf ( " cc -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].ccnd[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " prr1 -- "  );
  printf ( "\n" );
  for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].prr1[j] );
  }
  printf ( "\n" );
  printf ( " stt1 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %2.4f ", chn[0].stt1[i] );
  }
  printf ( "\n" );
  printf ( " q -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %2.4f ", chn[0].q[i] );
  }
  printf ( "\n" );
  printf ( " ru -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %2.4f ", chn[0].ru[i] );
  }
  printf ( "\n" );
  printf ( " xx0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].dim; i++ ) {
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.4f ", chn[0].xx0[i+j*chn[0].dim] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( " stt0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl/2; i++ ) {
    printf ( " %2.4f ", chn[0].stt0[i] );
  }
  printf ( "\n" );
  return 0;
}

__host__ int SpecData ( Cupar *cdp, const int verbose, Model *mdl, Spectrum *spc ) {
  float alpha = ALPHA, beta = BETA;
  int sumOfAllBins = 0;
  int sumOfSourceBins = 0;
  for ( int i = 0; i < NSPCTR; i++ ) {
    if ( verbose == 1 ) {
      printf ( ".................................................................\n" );
      printf ( " Spectrum number  -- %i\n", i );
      printf ( " Spectrum table   -- %s\n", spc[i].srcTbl );
      printf ( " ARF table        -- %s\n", spc[i].arfTbl );
      printf ( " RMF table        -- %s\n", spc[i].rmfTbl );
      printf ( " Background table -- %s\n", spc[i].bckgrndTbl );
    }
    ReadFitsData ( verbose, spc[i].srcTbl, spc[i].arfTbl, spc[i].rmfTbl, spc[i].bckgrndTbl, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfChnnls, spc[i].nmbrOfRmfVls, &spc[i].backscal_src, &spc[i].backscal_bkg, spc[i].srcCnts, spc[i].bckgrndCnts, spc[i].arfFctrs, spc[i].rmfVlsInCsc, spc[i].rmfIndxInCsc, spc[i].rmfPntrInCsc, spc[i].gdQltChnnls, spc[i].lwrChnnlBndrs, spc[i].hghrChnnlBndrs, spc[i].enrgChnnls, spc[i].nmbrOfBns, spc[i].grpVls, spc[i].grpIndx, spc[i].grpPntr, spc[i].grpng );
    int count = 0;
    while ( spc[i].lwrChnnlBndrs[spc[i].grpPntr[count]] < spc[i].lwrNtcdEnrg ) {
      count += 1;
    }
    spc[i].lwrBn = count;
    while ( spc[i].hghrChnnlBndrs[spc[i].grpPntr[count]] <= spc[i].hghrNtcdEnrg ) {
      count += 1;
    }
    spc[i].hghrBn = count - 1;
    spc[i].nmbrOfNtcdBns = spc[i].hghrBn - spc[i].lwrBn;
    spc[i].nmbrOfUsdBns = spc[i].hghrBn - spc[i].lwrBn;
    cudaMallocManaged ( ( void ** ) &spc[i].ntcIndx, spc[i].nmbrOfNtcdBns * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].ntcVls, spc[i].nmbrOfNtcdBns * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].ntcPntr, ( spc[i].nmbrOfNtcdBns + 1 ) * sizeof ( int ) );
    for ( int j = 0; j < spc[i].nmbrOfNtcdBns+1; j++ ) {
      spc[i].ntcPntr[j] = j;
    }
    for ( int j = 0; j < spc[i].nmbrOfNtcdBns; j++ ) {
      spc[i].ntcVls[j] = 1;
    }
    for ( int j = 0; j < spc[i].nmbrOfNtcdBns; j++ ) {
      spc[i].ntcIndx[j] = spc[i].lwrBn + j;
    }
    cudaMallocManaged ( ( void ** ) &spc[i].grpIgnPntr, ( spc[i].nmbrOfNtcdBns + 1 ) * sizeof ( int ) );
    cusparseXcsrgemmNnz ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, spc[i].nmbrOfChnnls, spc[i].nmbrOfBns, cdp[0].MatDescr, spc[i].nmbrOfNtcdBns, spc[i].ntcPntr, spc[i].ntcIndx, cdp[0].MatDescr, spc[i].nmbrOfChnnls, spc[i].grpPntr, spc[i].grpIndx, cdp[0].MatDescr, spc[i].grpIgnPntr, &spc[i].nmbrOfgrpIgnVls );
    //cudaDeviceSynchronize ();
    cudaMallocManaged ( ( void ** ) &spc[i].grpIgnIndx, spc[i].nmbrOfgrpIgnVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].grpIgnVls, spc[i].nmbrOfgrpIgnVls * sizeof ( float ) );
    cusparseScsrgemm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, spc[i].nmbrOfChnnls, spc[i].nmbrOfBns, cdp[0].MatDescr, spc[i].nmbrOfNtcdBns, spc[i].ntcVls, spc[i].ntcPntr, spc[i].ntcIndx, cdp[0].MatDescr, spc[i].nmbrOfChnnls, spc[i].grpVls, spc[i].grpPntr, spc[i].grpIndx, cdp[0].MatDescr, spc[i].grpIgnVls, spc[i].grpIgnPntr, spc[i].grpIgnIndx );
    cudaMallocManaged ( ( void ** ) &spc[i].srcGrp, spc[i].nmbrOfNtcdBns * sizeof ( float ) );
    cusparseScsrmv ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, spc[i].nmbrOfChnnls, spc[i].nmbrOfgrpIgnVls, &alpha, cdp[0].MatDescr, spc[i].grpIgnVls, spc[i].grpIgnPntr, spc[i].grpIgnIndx, spc[i].srcCnts, &beta, spc[i].srcGrp );
    cudaMallocManaged ( ( void ** ) &spc[i].bkgGrp, spc[i].nmbrOfNtcdBns * sizeof ( float ) );
    cusparseScsrmv ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, spc[i].nmbrOfChnnls, spc[i].nmbrOfgrpIgnVls, &alpha, cdp[0].MatDescr, spc[i].grpIgnVls, spc[i].grpIgnPntr, spc[i].grpIgnIndx, spc[i].bckgrndCnts, &beta, spc[i].bkgGrp );
    cusparseScsr2csc ( cdp[0].cusparseHandle, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfChnnls, spc[i].nmbrOfRmfVls, spc[i].rmfVlsInCsc, spc[i].rmfPntrInCsc, spc[i].rmfIndxInCsc, spc[i].rmfVls, spc[i].rmfIndx, spc[i].rmfPntr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO );
    cudaMallocManaged ( ( void ** ) &spc[i].iPntr, ( spc[i].nmbrOfNtcdBns + 1 ) * sizeof ( int ) );
    cusparseXcsrgemmNnz ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfBns, cdp[0].MatDescr, spc[i].nmbrOfgrpIgnVls, spc[i].grpIgnPntr, spc[i].grpIgnIndx, cdp[0].MatDescr, spc[i].nmbrOfRmfVls, spc[i].rmfPntr, spc[i].rmfIndx, cdp[0].MatDescr, spc[i].iPntr, &spc[i].nmbrOfiVls );
    //cudaDeviceSynchronize ();
    cudaMallocManaged ( ( void ** ) &spc[i].iIndx, spc[i].nmbrOfiVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].iVls, spc[i].nmbrOfiVls * sizeof ( float ) );
    cusparseScsrgemm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfBns, cdp[0].MatDescr, spc[i].nmbrOfgrpIgnVls, spc[i].grpIgnVls, spc[i].grpIgnPntr, spc[i].grpIgnIndx, cdp[0].MatDescr, spc[i].nmbrOfRmfVls, spc[i].rmfVls, spc[i].rmfPntr, spc[i].rmfIndx, cdp[0].MatDescr, spc[i].iVls, spc[i].iPntr, spc[i].iIndx );
    AssembleArrayOfPhotoelectricCrossections ( spc[i].nmbrOfEnrgChnnls, ATNMR, mdl[0].sgFlg, spc[i].enrgChnnls, mdl[0].atmcNmbrs, spc[i].crssctns );
    cudaDeviceSynchronize ( );
    sumOfAllBins += spc[i].nmbrOfNtcdBns;
    if ( i < NSPCTRCHI ) {
      sumOfSourceBins += spc[i].nmbrOfNtcdBns;
    }
    if ( verbose == 1 ) {
      printf ( " Number of energy channels                -- %i\n", spc[i].nmbrOfEnrgChnnls );
      printf ( " Number of instrument channels            -- %i\n", spc[i].nmbrOfChnnls );
      printf ( " Number of nonzero elements of RMF matrix -- %i\n", spc[i].nmbrOfRmfVls );
      printf ( " Number of grouping bins                  -- %i\n", spc[i].nmbrOfBns );
      printf ( " Number of noticed bins                   -- %i\n", spc[i].nmbrOfNtcdBns );
      printf ( " Exposure time                            -- %4.0f\n", spc[i].srcExptm );
      printf ( " Exposure time (background)               -- %4.0f\n", spc[i].bckgrndExptm );
      printf ( " Backscale src                            -- %4.0f\n", spc[i].backscal_src );
      printf ( " Backscale bkg                            -- %4.0f\n", spc[i].backscal_bkg );
    }
  }
  if ( verbose == 1 ) {
    printf ( ".................................................................\n" );
    printf ( " Total number of used data channels         -- %i\n", sumOfAllBins );
    printf ( " Number of used source channels             -- %i\n", sumOfSourceBins );
    printf ( " Total number of degrees of freedom         -- %i\n", sumOfAllBins - NPRS );
    printf ( " number of source degrees of freedom        -- %i\n", sumOfSourceBins - NPRS );
  }
  return 0;
}

__host__ int SpecInfo ( const char *spcLst[NSPCTR], const int verbose, Spectrum *spc ) {
  for ( int i = 0; i < NSPCTR; i++ ) {
    ReadFitsInfo ( spcLst[i], &spc[i].nmbrOfEnrgChnnls, &spc[i].nmbrOfChnnls, &spc[i].nmbrOfRmfVls, &spc[i].nmbrOfBns, &spc[i].srcExptm, &spc[i].bckgrndExptm, spc[i].srcTbl, spc[i].arfTbl, spc[i].rmfTbl, spc[i].bckgrndTbl );
  }
  return 0;
}

__host__ int SpecAlloc ( Chain *chn, Spectrum *spc ) {
  for ( int i = 0; i < NSPCTR; i++ ) {
    cudaMallocManaged ( ( void ** ) &spc[i].rmfPntrInCsc, ( spc[i].nmbrOfEnrgChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].rmfIndxInCsc, spc[i].nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].rmfPntr, ( spc[i].nmbrOfChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].rmfIndx, spc[i].nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].rmfVlsInCsc, spc[i].nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].rmfVls, spc[i].nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].enrgChnnls, ( spc[i].nmbrOfEnrgChnnls + 1 ) * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].arfFctrs, spc[i].nmbrOfEnrgChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].srcCnts, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].bckgrndCnts, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].lwrChnnlBndrs, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].hghrChnnlBndrs, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].gdQltChnnls, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].crssctns, spc[i].nmbrOfEnrgChnnls * ATNMR * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].absrptnFctrs, spc[i].nmbrOfEnrgChnnls * chn[0].nwl * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].mdlFlxs, spc[i].nmbrOfEnrgChnnls * chn[0].nwl * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].nsa1Flxs, ( spc[i].nmbrOfEnrgChnnls + 1 ) * chn[0].nwl * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].nsa2Flxs, ( spc[i].nmbrOfEnrgChnnls + 1 ) * chn[0].nwl * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].flddMdlFlxs, spc[i].nmbrOfChnnls * chn[0].nwl * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].ntcdChnnls, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].chnnlSttstcs, spc[i].nmbrOfChnnls * chn[0].nwl * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].grpng, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].grpPntr, ( spc[i].nmbrOfBns + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].grpIndx, spc[i].nmbrOfChnnls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spc[i].grpVls, spc[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spc[i].chiSttstcs, spc[i].nmbrOfChnnls * chn[0].nwl * sizeof ( float ) );
  }
  return 0;
}

__host__ void FreeSpec ( const Spectrum *spc ) {
  for ( int i = 0; i < NSPCTR; i++ ) {
    cudaFree ( spc[i].rmfVlsInCsc );
    cudaFree ( spc[i].rmfIndxInCsc );
    cudaFree ( spc[i].rmfPntrInCsc );
    cudaFree ( spc[i].rmfVls );
    cudaFree ( spc[i].rmfIndx );
    cudaFree ( spc[i].rmfPntr );
    cudaFree ( spc[i].enrgChnnls );
    cudaFree ( spc[i].arfFctrs );
    cudaFree ( spc[i].srcCnts );
    cudaFree ( spc[i].bckgrndCnts );
    cudaFree ( spc[i].gdQltChnnls );
    cudaFree ( spc[i].lwrChnnlBndrs );
    cudaFree ( spc[i].hghrChnnlBndrs );
    cudaFree ( spc[i].crssctns );
    cudaFree ( spc[i].absrptnFctrs );
    cudaFree ( spc[i].mdlFlxs );
    cudaFree ( spc[i].nsa1Flxs );
    cudaFree ( spc[i].nsa2Flxs );
    cudaFree ( spc[i].flddMdlFlxs );
    cudaFree ( spc[i].chnnlSttstcs );
    cudaFree ( spc[i].ntcdChnnls );
    cudaFree ( spc[i].grpVls );
    cudaFree ( spc[i].grpIndx );
    cudaFree ( spc[i].grpPntr );
    cudaFree ( spc[i].grpng );
    cudaFree ( spc[i].ntcIndx );
    cudaFree ( spc[i].ntcPntr );
    cudaFree ( spc[i].ntcVls );
    cudaFree ( spc[i].grpIgnVls );
    cudaFree ( spc[i].grpIgnIndx );
    cudaFree ( spc[i].grpIgnPntr );
    cudaFree ( spc[i].iVls );
    cudaFree ( spc[i].iIndx );
    cudaFree ( spc[i].iPntr );
    cudaFree ( spc[i].bkgGrp );
    cudaFree ( spc[i].srcGrp );
    cudaFree ( spc[i].chiSttstcs );
  }
}

__host__ int ReadFitsInfo ( const char *spcFl, int *nmbrOfEnrgChnnls, int *nmbrOfChnnls, int *nmbrOfRmfVls, int *nmbrOfBins, float *srcExptm, float *bckgrndExptm, char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] ) {
  fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
  int status = 0, intnull = 0, anynull = 0, colnum;
  char card[FLEN_CARD], colNgr[] = "N_GRP", colGrp[] = "GROUPING", colNch[] = "N_CHAN";
  float floatnull;
  /* Open Spectrum  */
  snprintf ( srcTbl, sizeof ( card ), "%s%s", spcFl, "[SPECTRUM]" );
  fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
  fits_read_key ( ftsPntr, TINT, "NAXIS2", nmbrOfChnnls, NULL, &status );
  fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", srcExptm, NULL, &status );
  /* Read names of arf rmf and background */
  fits_read_key ( ftsPntr, TSTRING, "ANCRFILE", card, NULL, &status );
  snprintf ( arfTbl, sizeof ( card ), "%s%s", card, "[SPECRESP]" );
  fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
  snprintf ( rmfTbl, sizeof ( card ), "%s%s", card, "[MATRIX]" );
  /* Open Background file */
  fits_read_key ( ftsPntr, TSTRING, "BACKFILE", card, NULL, &status );
  snprintf ( bckgrndTbl, sizeof ( card ), "%s%s", card, "[SPECTRUM]" );
  int *grp;
  grp = ( int * ) malloc ( *nmbrOfChnnls * sizeof ( int ) );
  fits_get_colnum ( ftsPntr, CASEINSEN, colGrp, &colnum, &status );
  fits_read_col_int ( ftsPntr, colnum, 1, 1, *nmbrOfChnnls, intnull, grp, &anynull, &status );
  int sum = 0;
  for ( int i = 0; i < *nmbrOfChnnls; i++ ) {
    sum += ( grp[i] > 0 );
  }
  *nmbrOfBins = sum;
  fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
  if ( status == 0 && BACKIN == 1 ) {
    fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", bckgrndExptm, NULL, &status );
  } else {
    *bckgrndExptm = 0.0;
    status = 0;
  }
  /* Open RMF file */
  fits_open_file ( &ftsPntr, rmfTbl, READONLY, &status );
  if ( status != 0 ) { printf ( " Error: Opening rmf table fails\n" ); return 1; }
  fits_read_key ( ftsPntr, TINT, "NAXIS2", nmbrOfEnrgChnnls, NULL, &status );
  if ( status != 0 ) { printf ( " Error: Reading NAXIS2 key from rmf table fails\n" ); return 1; }
  int *n_grp;
  n_grp = ( int * ) malloc ( *nmbrOfEnrgChnnls * sizeof ( int ) );
  fits_get_colnum ( ftsPntr, CASEINSEN, colNgr, &colnum, &status );
  fits_read_col_int ( ftsPntr, colnum, 1, 1, *nmbrOfEnrgChnnls, intnull, n_grp, &anynull, &status );
  int *n_chan_vec;
  n_chan_vec = ( int * ) malloc ( *nmbrOfChnnls * sizeof ( int ) );
  sum = 0;
  for ( int i = 0; i < *nmbrOfEnrgChnnls; i++ ) {
    fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
    fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, n_chan_vec, &anynull, &status );
    for ( int j = 0; j < n_grp[i]; j++ ) {
      sum = sum + n_chan_vec[j];
    }
  }
  *nmbrOfRmfVls = sum;
  free ( n_chan_vec );
  free ( n_grp );
  free ( grp );
  return 0;
}

__host__ int ReadFitsData ( const int verbose, const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD], const int nmbrOfEnrgChnnls, const int nmbrOfChnnls, const int nmbrOfRmfVls, float *backscal_src, float *backscal_bkg, float *srcCnts, float *bckgrndCnts, float *arfFctrs, float *rmfVlsInCsc, int *rmfIndxInCsc, int *rmfPntrInCsc, float *gdQltChnnls, float *lwrChnnlBndrs, float *hghrChnnlBndrs, float *enrgChnnls, const int nmbrOfBns, float *grpVls, int *grpIndx, int *grpPntr, float *grpng ) {
  fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
  int status = 0, anynull, colnum, intnull = 0, rep_chan = 100;
  char card[FLEN_CARD], EboundsTable[FLEN_CARD], Telescop[FLEN_CARD];
  char colNgr[]="N_GRP", colNch[]="N_CHAN",  colFch[]="F_CHAN", colCounts[]="COUNTS", colSpecResp[]="SPECRESP", colEnLo[]="ENERG_LO", colEnHi[]="ENERG_HI", colMat[]="MATRIX", colEmin[]="E_MIN", colEmax[]="E_MAX", colGrp[] = "GROUPING";
  float floatnull;
  /* Read Spectrum: */
  fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
  fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
  snprintf ( EboundsTable, sizeof ( EboundsTable ), "%s%s", card, "[EBOUNDS]" );
  fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", backscal_src, NULL, &status );
  fits_read_key ( ftsPntr, TSTRING, "TELESCOP", Telescop, NULL, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, srcCnts, &anynull, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colGrp, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, grpng, &anynull, &status );
  int count = 0;
  for ( int j = 0; j < nmbrOfChnnls; j++ ) {
    grpVls[j] = 1;
    grpIndx[j] = j;
    if ( grpng[j] == 1 ) {
      grpPntr[count] = j;
      count += 1;
    }
  }
  grpPntr[count] = nmbrOfChnnls;
  /* Read ARF FILE: */
  fits_open_file ( &ftsPntr, arfTbl, READONLY, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colSpecResp, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, arfFctrs, &anynull, &status );
  /* Read Background: */
  fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
  if ( status == 0 && BACKIN == 1 ) {
    fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", backscal_bkg, NULL, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, bckgrndCnts, &anynull, &status );
  } else {
    if ( verbose == 1) {
      printf ( " Warning: Background table is not used, background exposure and background are set to 0.\n " );
    }
    for ( int i = 0; i < nmbrOfChnnls; i++ ) {
      bckgrndCnts[i] = 0;
    }
    status = 0;
  }
  /* Read RMF file */
  fits_open_file ( &ftsPntr, rmfTbl, READONLY, &status );
  float *enelo_vec, *enehi_vec;
  enelo_vec = ( float * ) malloc ( nmbrOfEnrgChnnls * sizeof ( float ) );
  enehi_vec = ( float * ) malloc ( nmbrOfEnrgChnnls * sizeof ( float ) );
  fits_get_colnum ( ftsPntr, CASEINSEN, colEnLo, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, enelo_vec, &anynull, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colEnHi, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, enehi_vec, &anynull, &status );
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
    enrgChnnls[i] = enelo_vec[i];
  }
  enrgChnnls[nmbrOfEnrgChnnls] = enehi_vec[nmbrOfEnrgChnnls-1];
  int *f_chan_vec, *n_chan_vec;
  f_chan_vec = ( int * ) malloc ( rep_chan * sizeof ( int ) );
  n_chan_vec = ( int * ) malloc ( rep_chan * sizeof ( int ) );
  int *f_chan, *n_chan;
  f_chan = ( int * ) malloc ( rep_chan * nmbrOfEnrgChnnls * sizeof ( int ) );
  n_chan = ( int * ) malloc ( rep_chan * nmbrOfEnrgChnnls * sizeof ( int ) );
  int *n_grp;
  n_grp = ( int * ) malloc ( nmbrOfEnrgChnnls * sizeof ( int ) );
  fits_get_colnum ( ftsPntr, CASEINSEN, colNgr, &colnum, &status );
  fits_read_col_int ( ftsPntr, colnum, 1, 1, nmbrOfEnrgChnnls, intnull, n_grp, &anynull, &status );
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
    fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
    fits_read_col_int ( ftsPntr, colnum, i+1, 1, n_grp[i], intnull, n_chan_vec, &anynull, &status );
    for ( int j = 0; j < rep_chan; j++ ) {
      n_chan[i*rep_chan+j] = n_chan_vec[j];
    }
  }
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
    fits_get_colnum ( ftsPntr, CASEINSEN, colFch, &colnum, &status );
    fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, f_chan_vec, &anynull, &status );
    for ( int j = 0; j < rep_chan; j++ ) {
      f_chan[i*rep_chan+j] = f_chan_vec[j];
    }
  }
  int sum = 0;
  rmfPntrInCsc[0] = 0;
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
    for ( int j = 0; j < n_grp[i]; j++ ) {
      sum = sum + n_chan[rep_chan*i+j];
    }
    rmfPntrInCsc[i+1] = sum;
  }
  int m = 0;
  if ( nmbrOfChnnls != 1024 ) {
    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
      for ( int j = 0; j < n_grp[i]; j++ ) {
        for ( int k = f_chan[rep_chan*i+j] ; k < f_chan[rep_chan*i+j] + n_chan[rep_chan*i+j]; k++ ) {
          rmfIndxInCsc[m] = k;
          m = m + 1;
        }
      }
    }
  } else if ( nmbrOfChnnls == 1024 ) {
    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
      for ( int j = 0; j < n_grp[i]; j++ ) {
        for ( int k = f_chan[rep_chan*i+j] - 1; k < f_chan[rep_chan*i+j] - 1 + n_chan[rep_chan*i+j]; k++ ) {
          rmfIndxInCsc[m] = k;
          m = m + 1;
        }
      }
    }
  }
  float *rmf_vec;
  rmf_vec = ( float * ) malloc ( nmbrOfChnnls * sizeof ( float ) );
  fits_get_colnum ( ftsPntr, CASEINSEN, colMat, &colnum, &status );
  m = 0;
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
    sum = rmfPntrInCsc[i+1] - rmfPntrInCsc[i];
    fits_read_col ( ftsPntr, TFLOAT, colnum, i+1, 1, sum, &floatnull, rmf_vec, &anynull, &status );
    for ( int k = 0; k < sum; k++ ) {
      rmfVlsInCsc[m] = rmf_vec[k];
      m = m + 1;
    }
  }
  /* Read Ebounds Table: */
  fits_open_file ( &ftsPntr, EboundsTable, READONLY, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colEmin, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, lwrChnnlBndrs, &anynull, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colEmax, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, hghrChnnlBndrs, &anynull, &status );
  free ( enelo_vec );
  free ( enehi_vec );
  free ( rmf_vec );
  free ( f_chan_vec );
  free ( n_chan_vec );
  free ( n_chan );
  free ( f_chan );
  free ( n_grp );
  return 0;
}

__host__ void FreeModel ( const Model *mdl ) {
  cudaFree ( mdl[0].atmcNmbrs );
  cudaFree ( mdl[0].abndncs );
  cudaFree ( mdl[0].RedData );
  cudaFree ( mdl[0].Dist );
  cudaFree ( mdl[0].EBV );
  cudaFree ( mdl[0].errDist );
  cudaFree ( mdl[0].errEBV );
  cudaFree ( mdl[0].RedData1 );
  cudaFree ( mdl[0].Dist1 );
  cudaFree ( mdl[0].EBV1 );
  cudaFree ( mdl[0].RedData2 );
  cudaFree ( mdl[0].Dist2 );
  cudaFree ( mdl[0].EBV2 );
  cudaFree ( mdl[0].RedData3 );
  cudaFree ( mdl[0].Dist3 );
  cudaFree ( mdl[0].EBV3 );
  cudaFree ( mdl[0].nsaDt );
  cudaFree ( mdl[0].nsaT );
  cudaFree ( mdl[0].nsaE );
  cudaFree ( mdl[0].nsaFlxs );
  cudaFree ( mdl[0].nsmaxgDt );
  cudaFree ( mdl[0].nsmaxgT );
  cudaFree ( mdl[0].nsmaxgE );
  cudaFree ( mdl[0].nsmaxgFlxs );
}

__global__ void BilinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int tIndx, const int grIndx, const float *data, const float *xin, const float *yin, const int M1, const int M2, const float *enrgChnnls, const float *wlkrs, float *mdlFlxs ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  float xxout, yyout, sa, gr, a, b, d00, d01, d10, d11, tmp1, tmp2, tmp3;
  int v, w;
  if ( i < nmbrOfEnrgChnnls && j < nmbrOfWlkrs ) {
    gr = sqrtf ( 1. - 2.952 * MNS / RNS );
    sa = powf ( RNS, 2. );
    xxout = log10f ( enrgChnnls[i] / gr );
    yyout = wlkrs[tIndx+j*NPRS];
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
    mdlFlxs[i+j*nmbrOfEnrgChnnls] = powf ( 10., tmp3 ) * sa;
  }
}

__global__ void BilinearInterpolationNsmax ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int tIndx, const int grIndx, const float *data, const float *xin, const float *yin, const int M1, const int M2, const float *enrgChnnls, const float *wlkrs, float *mdlFlxs ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  float xxout, yyout, sa, gr, a, b, d00, d01, d10, d11, tmp1, tmp2, tmp3;
  int v, w;
  if ( i < nmbrOfEnrgChnnls && j < nmbrOfWlkrs ) {
    gr = sqrtf ( 1. - 2.952 * MNS / RNS );
    sa = powf ( RNS, 2. );
    xxout = log10f ( enrgChnnls[i] / gr );
    yyout = wlkrs[tIndx+j*NPRS];
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
    mdlFlxs[i+j*nmbrOfEnrgChnnls] = powf ( 10., tmp3 + 26.1787440 - xxout ) * sa;
  }
}

__global__ void LinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int dIndx, const float *Dist, const float *EBV, const float *errEBV, const float *wlkrs, float *mNh, float *sNh )
{
  int w = threadIdx.x + blockDim.x * blockIdx.x;
  float xxout, a, dmNh0, dmNh1, dsNh0, dsNh1, tmpMNh, tmpSNh;
  int v;
  if ( w < nmbrOfWlkrs )
  {
    xxout = wlkrs[dIndx+w*NPRS];
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

__global__ void LinearInterpolationNoErrors ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int dIndx, const float *Dist, const float *EBV, const float *wlkrs, float *mNh, float *sNh )
{
  int w = threadIdx.x + blockDim.x * blockIdx.x;
  float xxout, a, dmNh0, dmNh1, tmpMNh;
  int v;
  if ( w < nmbrOfWlkrs )
  {
    xxout = wlkrs[dIndx+w*NPRS];
    v = FindElementIndex ( Dist, nmbrOfDistBins, xxout );
    a = ( xxout - Dist[v] ) / ( Dist[v+1] - Dist[v] );
    if ( v < nmbrOfDistBins ) dmNh0 = EBV[v]; else dmNh0 = 0;
    if ( v+1 < nmbrOfDistBins ) dmNh1 = EBV[v+1]; else dmNh1 = 0;
    tmpMNh = a * dmNh1 + ( -dmNh0 * a + dmNh0 );
    tmpMNh = powf ( 10, tmpMNh );
    mNh[w] = 0.7 * tmpMNh;
    sNh[w] = 0.7 * tmpMNh * 0.1;
  }
}

__global__ void ReverseLinearInterpolationNoErrors ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int dIndx, const float *Dist, const float *EBV, const float *wlkrs, float *dist ) {
  int w = threadIdx.x + blockDim.x * blockIdx.x;
  float xxout, a, dmNh0, dmNh1, tmpMNh;
  int v;
  if ( w < nmbrOfWlkrs ) {
    xxout = wlkrs[NHINDX+w*NPRS] / 0.7;
    v = FindElementIndex ( EBV, nmbrOfDistBins, xxout );
    a = ( xxout - EBV[v] ) / ( EBV[v+1] - EBV[v] );
    if ( v < nmbrOfDistBins ) dmNh0 = Dist[v]; else dmNh0 = 0;
    if ( v+1 < nmbrOfDistBins ) dmNh1 = Dist[v+1]; else dmNh1 = 0;
    tmpMNh = a * dmNh1 + ( -dmNh0 * a + dmNh0 );
    //tmpMNh = powf ( 10, tmpMNh );
    dist[w] = tmpMNh;
  }
}

__global__ void AssembleArrayOfModelFluxes ( const int spIndx, const int nwl, const int nmbrOfEnrgChnnls, const float backscal_src, const float backscal_bkg, const float *en, const float *arf, const float *absrptn, const float *wlk, const float *nsa1Flx, float *flx, const float *didi ) {
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  int w = threadIdx.y + blockDim.y * blockIdx.y;
  int t = e + w * nmbrOfEnrgChnnls;
  float f = 0, Norm, intNsaFlx;
  float scl = backscal_src / backscal_bkg;
  if ( e < nmbrOfEnrgChnnls && w < nwl ) {
    if ( spIndx == 0 ) {
      //intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      intNsaFlx = BlackBody ( wlk[0+w*NPRS], en[e], en[e+1] );//PowerLaw ( wlk[0+w*NPRS], wlk[1+w*NPRS], en[e], en[e+1] )
      Norm = powf ( 10., - 2 * didi[w] + 2 * wlk[1+w*NPRS] + 2 * KMCMPCCM ); //
      f += Norm * intNsaFlx;
      f += PowerLaw ( wlk[4+w*NPRS], wlk[5+w*NPRS], en[e], en[e+1] );
      f *= absrptn[t];
      f += scl * PowerLaw ( wlk[8+w*NPRS], wlk[9+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 1 ) {
      //intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      intNsaFlx = BlackBody ( wlk[0+w*NPRS], en[e], en[e+1] );
      Norm = powf ( 10., - 2 * didi[w] + 2 * wlk[2+w*NPRS] + 2 * KMCMPCCM ); //
      f += Norm * intNsaFlx;
      f += PowerLaw ( wlk[4+w*NPRS], wlk[5+w*NPRS], en[e], en[e+1] );
      f *= absrptn[t];
      f += scl * PowerLaw ( wlk[10+w*NPRS], wlk[11+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 2 ) {
      //intNsaFlx = IntegrateNsa ( nsa1Flx[e+w*(nmbrOfEnrgChnnls+1)], nsa1Flx[e+1+w*(nmbrOfEnrgChnnls+1)], en[e], en[e+1] );
      intNsaFlx = BlackBody ( wlk[0+w*NPRS], en[e], en[e+1] );
      Norm = powf ( 10., - 2 * didi[w] + 2 * wlk[3+w*NPRS] + 2 * KMCMPCCM ); //
      f += Norm * intNsaFlx;
      f += PowerLaw ( wlk[4+w*NPRS], wlk[5+w*NPRS], en[e], en[e+1] );
      f *= absrptn[t];
      f += scl * PowerLaw ( wlk[12+w*NPRS], wlk[13+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 3 ) {
      f += PowerLaw ( wlk[6+w*NPRS], wlk[7+w*NPRS], en[e], en[e+1] );
      f *= absrptn[t];
      f += scl * PowerLaw ( wlk[14+w*NPRS], wlk[15+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 4 ) {
      f += PowerLaw ( wlk[6+w*NPRS], wlk[7+w*NPRS], en[e], en[e+1] );
      f *= absrptn[t];
      f += scl * PowerLaw ( wlk[16+w*NPRS], wlk[17+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 5 ) {
      f += PowerLaw ( wlk[6+w*NPRS], wlk[7+w*NPRS], en[e], en[e+1] );
      f *= absrptn[t];
      f += scl * PowerLaw ( wlk[18+w*NPRS], wlk[19+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 6 ) {
      f += PowerLaw ( wlk[8+w*NPRS], wlk[9+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 7 ) {
      f += PowerLaw ( wlk[10+w*NPRS], wlk[11+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 8 ) {
      f += PowerLaw ( wlk[12+w*NPRS], wlk[13+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 9 ) {
      f += PowerLaw ( wlk[14+w*NPRS], wlk[15+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 10 ) {
      f += PowerLaw ( wlk[16+w*NPRS], wlk[17+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    } else if ( spIndx == 11 ) {
      f += PowerLaw ( wlk[18+w*NPRS], wlk[19+w*NPRS], en[e], en[e+1] );
      f *= arf[e];
      flx[t] = f;
    }
  }
}

__host__ int modelStatistic1 ( const Cupar *cdp, const Model *mdl, Chain *chn, Spectrum *spc ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA, beta1 = 1.;
  constantArray <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].nwl/2, 0., chn[0].stt1 );
  constantArray <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].nwl/2, 0., chn[0].chi1 );
  for ( int i = 0; i < NSPCTR; i++ ) {
    AssembleArrayOfAbsorptionFactors <<< grid2D ( spc[i].nmbrOfEnrgChnnls, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls, ATNMR, spc[i].crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].xx1, spc[i].absrptnFctrs );
    //BilinearInterpolationNsmax <<< grid2D ( spc[i].nmbrOfEnrgChnnls+1, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls+1, 0, GRINDX, mdl[0].nsmaxgFlxs, mdl[0].nsmaxgE, mdl[0].nsmaxgT, mdl[0].numNsmaxgE, mdl[0].numNsmaxgT, spc[i].enrgChnnls, chn[0].xx1, spc[i].nsa1Flxs );
    //BilinearInterpolation <<< grid2D ( spc[i].nmbrOfEnrgChnnls+1, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls+1, 0, GRINDX, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spc[i].enrgChnnls, chn[0].xx1, spc[i].nsa1Flxs );
    ReverseLinearInterpolationNoErrors <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>>  ( chn[0].nwl/2, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist1, mdl[0].EBV1, chn[0].xx1, chn[0].didi11 );
    ReverseLinearInterpolationNoErrors <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>>  ( chn[0].nwl/2, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist2, mdl[0].EBV2, chn[0].xx1, chn[0].didi12 );
    ReverseLinearInterpolationNoErrors <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>>  ( chn[0].nwl/2, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist3, mdl[0].EBV3, chn[0].xx1, chn[0].didi13 );
    chooseLaw <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].kex, chn[0].didi11, chn[0].didi12, chn[0].didi13, chn[0].didi1 );
    AssembleArrayOfModelFluxes <<< grid2D ( spc[i].nmbrOfEnrgChnnls, chn[0].nwl/2 ), block2D () >>> ( i, chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].enrgChnnls, spc[i].arfFctrs, spc[i].absrptnFctrs, chn[0].xx1, spc[i].nsa1Flxs, spc[i].mdlFlxs, chn[0].didi1 );
    cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfiVls, &alpha, cdp[0].MatDescr, spc[i].iVls, spc[i].iPntr, spc[i].iIndx, spc[i].mdlFlxs, spc[i].nmbrOfEnrgChnnls, &beta, spc[i].flddMdlFlxs, spc[i].nmbrOfNtcdBns );
    //AssembleArrayOfChannelStatistics <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].bckgrndExptm, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].srcGrp, spc[i].bkgGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    arrayOfCStat <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    //arrayOfChiSquareds <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    //arrayOfChiSquaredsWithBackground <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].bkgGrp, spc[i].backscal_src/spc[i].backscal_bkg, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfNtcdBns, chn[0].nwl/2, &alpha, spc[i].chnnlSttstcs, spc[i].nmbrOfNtcdBns, spc[i].grpVls, INCXX, &beta1, chn[0].stt1, INCYY );
    if ( i < NSPCTRCHI ) {
      arrayOfChiSquareds <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].flddMdlFlxs, spc[i].chiSttstcs );
      cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfNtcdBns, chn[0].nwl/2, &alpha, spc[i].chiSttstcs, spc[i].nmbrOfNtcdBns, spc[i].grpVls, INCXX, &beta1, chn[0].chi1, INCYY );
    }
    /*cudaDeviceSynchronize ();
    printf ( "Model --\n" );
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      for ( int k = 0; k < spc[i].nmbrOfNtcdBns; k++ ) {
        printf ( " %2.0f ", spc[i].chiSttstcs[k+j*spc[i].nmbrOfNtcdBns] );
      }
      printf("\n");
      for ( int k = 0; k < spc[i].nmbrOfNtcdBns; k++ ) {
        printf ( " %2.0f ", spc[i].flddMdlFlxs[k+j*spc[i].nmbrOfNtcdBns]*spc[i].srcExptm );
      }
      printf("\n");
      for ( int k = 0; k < spc[i].nmbrOfNtcdBns; k++ ) {
        printf ( " %2.0f ", spc[i].srcGrp[k] );
      }
      printf("\n");
      for ( int k = 0; k < spc[i].nmbrOfNtcdBns; k++ ) {
        printf ( " %2.0f ", powf ( spc[i].srcGrp[k] - spc[i].flddMdlFlxs[k+j*spc[i].nmbrOfNtcdBns]*spc[i].srcExptm, 2. ) / spc[i].srcGrp[k] );
      }
      printf("\n");
      printf("\n");
      printf("\n");
      printf("\n");
      printf("\n");
      printf("\n");
    }
    printf ("\n");
    printf ( "Chi --\n" );
    for ( int j = 0; j < chn[0].nwl/2; j++ ) {
      printf ( " %2.0f ", chn[0].chi1[j] );
    }
    printf ("\n");*/
    /*cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfChnnls, chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfRmfVls, &alpha, cdp[0].MatDescr, spc[i].rmfVls, spc[i].rmfPntr, spc[i].rmfIndx, spc[i].mdlFlxs, spc[i].nmbrOfEnrgChnnls, &beta, spc[i].flddMdlFlxs, spc[i].nmbrOfChnnls );
    AssembleArrayOfChannelStatistics <<< grid2D ( spc[i].nmbrOfChnnls, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfChnnls, spc[i].srcExptm, spc[i].bckgrndExptm, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].srcCnts, spc[i].bckgrndCnts, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfChnnls, chn[0].nwl/2, &alpha, spc[i].chnnlSttstcs, spc[i].nmbrOfChnnls, spc[i].grpVls, INCXX, &beta1, chn[0].stt1, INCYY );*/
  }
  arrayOf2DConditions <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xbnd, chn[0].xx1, chn[0].ccnd );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl/2, &alpha, chn[0].ccnd, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].cnd, incyy );
  //arrayOfPriors  <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].dim, chn[0].nwl/2, chn[0].cnd, chn[0].xx1, chn[0].prr1 );
  LinearInterpolationNoErrors <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].nwl/2, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist1, mdl[0].EBV1, chn[0].xx1, chn[0].nhMd, chn[0].nhSg );
  arrayOfPriors1 <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].dim, chn[0].nwl/2, chn[0].cnd, chn[0].nhMd, chn[0].nhSg, chn[0].xx1, chn[0].prr1 );
  return 0;
}

__host__ int modelStatistic0 ( const Cupar *cdp, const Model *mdl, Chain *chn, Spectrum *spc ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA, beta1 = 1.;
  for ( int i = 0; i < NSPCTR; i++ ) {
    AssembleArrayOfAbsorptionFactors <<< grid2D ( spc[i].nmbrOfEnrgChnnls, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfEnrgChnnls, ATNMR, spc[i].crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].xx, spc[i].absrptnFctrs );
    //BilinearInterpolationNsmax <<< grid2D ( spc[i].nmbrOfEnrgChnnls+1, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfEnrgChnnls+1, 0, GRINDX, mdl[0].nsmaxgFlxs, mdl[0].nsmaxgE, mdl[0].nsmaxgT, mdl[0].numNsmaxgE, mdl[0].numNsmaxgT, spc[i].enrgChnnls, chn[0].xx, spc[i].nsa1Flxs );
    //BilinearInterpolation <<< grid2D ( spc[i].nmbrOfEnrgChnnls+1, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfEnrgChnnls+1, 0, GRINDX, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spc[i].enrgChnnls, chn[0].xx, spc[i].nsa1Flxs );
    ReverseLinearInterpolationNoErrors <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>>  ( chn[0].nwl, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist1, mdl[0].EBV1, chn[0].xx, chn[0].didi01 );
    ReverseLinearInterpolationNoErrors <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>>  ( chn[0].nwl, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist2, mdl[0].EBV2, chn[0].xx, chn[0].didi02 );
    ReverseLinearInterpolationNoErrors <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>>  ( chn[0].nwl, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist3, mdl[0].EBV3, chn[0].xx, chn[0].didi03 );
    chooseLaw <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].kex, chn[0].didi01, chn[0].didi02, chn[0].didi03, chn[0].didi );
    AssembleArrayOfModelFluxes <<< grid2D ( spc[i].nmbrOfEnrgChnnls, chn[0].nwl ), block2D () >>> ( i, chn[0].nwl, spc[i].nmbrOfEnrgChnnls, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].enrgChnnls, spc[i].arfFctrs, spc[i].absrptnFctrs, chn[0].xx, spc[i].nsa1Flxs, spc[i].mdlFlxs, chn[0].didi );
    cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, chn[0].nwl, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfiVls, &alpha, cdp[0].MatDescr, spc[i].iVls, spc[i].iPntr, spc[i].iIndx, spc[i].mdlFlxs, spc[i].nmbrOfEnrgChnnls, &beta, spc[i].flddMdlFlxs, spc[i].nmbrOfNtcdBns );
    //AssembleArrayOfChannelStatistics <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].bckgrndExptm, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].srcGrp, spc[i].bkgGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    arrayOfCStat <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    //arrayOfChiSquareds <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    //arrayOfChiSquaredsWithBackground <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].bkgGrp, spc[i].backscal_src/spc[i].backscal_bkg, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfNtcdBns, chn[0].nwl, &alpha, spc[i].chnnlSttstcs, spc[i].nmbrOfNtcdBns, spc[i].grpVls, INCXX, &beta1, chn[0].stt, INCYY );
    if ( i < NSPCTRCHI ) {
      arrayOfChiSquareds <<< grid2D ( spc[i].nmbrOfNtcdBns, chn[0].nwl ), block2D () >>> ( chn[0].nwl, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].flddMdlFlxs, spc[i].chiSttstcs );
      cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfNtcdBns, chn[0].nwl, &alpha, spc[i].chiSttstcs, spc[i].nmbrOfNtcdBns, spc[i].grpVls, INCXX, &beta1, chn[0].chi, INCYY );
    }
    /*cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfChnnls, chn[0].nwl/2, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfRmfVls, &alpha, cdp[0].MatDescr, spc[i].rmfVls, spc[i].rmfPntr, spc[i].rmfIndx, spc[i].mdlFlxs, spc[i].nmbrOfEnrgChnnls, &beta, spc[i].flddMdlFlxs, spc[i].nmbrOfChnnls );
    AssembleArrayOfChannelStatistics <<< grid2D ( spc[i].nmbrOfChnnls, chn[0].nwl/2 ), block2D () >>> ( chn[0].nwl/2, spc[i].nmbrOfChnnls, spc[i].srcExptm, spc[i].bckgrndExptm, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].srcCnts, spc[i].bckgrndCnts, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfChnnls, chn[0].nwl, &alpha, spc[i].chnnlSttstcs, spc[i].nmbrOfChnnls, spc[i].grpVls, INCXX, &beta1, chn[0].stt, INCYY );*/
  }
  arrayOf2DConditions <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xbnd, chn[0].xx, chn[0].ccnd );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl, &alpha, chn[0].ccnd, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].cnd, incyy );
  //arrayOfPriors  <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].dim, chn[0].nwl, chn[0].cnd, chn[0].xx, chn[0].prr );
  LinearInterpolationNoErrors <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, mdl[0].nmbrOfDistBins1, DINDX1, mdl[0].Dist1, mdl[0].EBV1, chn[0].xx, chn[0].nhMd, chn[0].nhSg );
  arrayOfPriors1 <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].dim, chn[0].nwl, chn[0].cnd, chn[0].nhMd, chn[0].nhSg, chn[0].xx, chn[0].prr );
  //constantArray <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, 0., chn[0].prr );
  return 0;
}

__host__ __device__ float PowerLaw ( const float phtnIndx, const float nrmlztn, const float enrgLwr, const float enrgHghr ) {
  float flx;
  if ( fabsf ( 1 - phtnIndx ) > TLR ) {
    flx = powf ( 10, nrmlztn ) * ( powf ( enrgHghr, 1 - phtnIndx ) - powf ( enrgLwr, 1 - phtnIndx ) ) / ( 1 - phtnIndx );
  } else {
    flx = powf ( 10, nrmlztn ) * ( logf ( enrgHghr ) - logf ( enrgLwr ) );
  }
  return flx;
}

__host__ __device__ float IntegrateNsa ( const float flx1, const float flx2, const float en1, const float en2 ) {
  float flx;
  flx = 0.5 * ( flx1 + flx2 ) * ( en2 - en1 );
  return flx;
}

__host__ __device__ float IntegrateNsmax ( const float flx1, const float flx2, const float en1, const float en2 ) {
  float flx;
  float gr = sqrtf ( 1. - 2.952 * MNS / RNS );
  flx = gr * 0.5 * ( flx1 * ( en2/en1 - 1 ) + flx2 * ( 1 - en1/en2 ) );
  return flx;
}

__host__ __device__ float BlackBody ( const float Teff, const float enrgLwr, const float enrgHghr ) {
  float t, anorm, elow, x, tinv, anormh, alow, ehi, ahi, flx;
  float kb = 1.38E-16;
  float gr = sqrtf ( 1. - 2.952 * MNS / RNS );
  t = Teff; //powf ( 10., Teff ) * kb / 1.6022E-9 * gr;
  tinv = 1. / t;
  anorm = 1.0344e-3f * powf ( RNS, 2. ) * 1.e8f; //  * powf ( 10, logRtD ) ;//* 1e8f
  anormh = 0.5 * anorm;
  elow = enrgLwr;
  x = elow * tinv;
  if ( x <= 1.0e-4f ) {
    alow = elow * t;
  }
  else if ( x > 60.0 ) {
    flx = 0;
    return flx;
  } else {
    alow = elow * elow / ( expf ( x ) - 1.0e0f );
  }
  ehi = enrgHghr;
  x = ehi * tinv;
  if ( x <= 1.0e-4f ) {
    ahi = ehi * t;
  } else if ( x > 60.0 ) {
    flx = 0;
    return flx;
  } else {
    ahi = ehi * ehi / ( expf ( x ) - 1.0e0f );
  }
  flx = anormh * ( alow + ahi ) * ( ehi - elow );
  return flx;
}

__host__ __device__ float Poisson ( const float scnts, const float mdl, const float ts ) {
  float sttstc = 0;
  if ( scnts != 0 && ts * mdl >= TLR ) {
    sttstc = ts * mdl - scnts * logf ( ts * mdl ) - scnts * ( 1 - logf ( scnts ) );
  } else if ( scnts != 0 && ts * mdl < TLR ) {
    sttstc = TLR - scnts * logf ( TLR ) - scnts * ( 1 - logf ( scnts ) );
  } else {
    sttstc = ts * mdl;
  }
  sttstc = 2 * sttstc;
  return sttstc;
}

__host__ __device__ float PoissonWithBackground ( const float scnts, const float bcnts, const float mdl, const float ts, const float tb, const float backscal_src, const float backscal_bkg ) {
  float sttstc = 0, d, f;
  float scls = 1;
  float sclb = backscal_bkg / backscal_src;
  d = sqrtf ( powf ( ( ts * scls + tb * sclb ) * mdl - scnts - bcnts, 2. ) + 4 * ( ts * scls + tb * sclb ) * bcnts * mdl );
  f = ( scnts + bcnts - ( ts * scls + tb * sclb ) * mdl + d ) / 2 / ( ts * scls + tb * sclb );
  if ( scnts != 0 && bcnts != 0 ) {
    sttstc = ts * mdl + ts * scls * f  + tb * sclb * f - scnts * logf ( ts * mdl + ts * scls * f ) - bcnts * logf ( tb * sclb * f ) - scnts * ( 1 - logf ( scnts ) ) - bcnts * ( 1 - logf ( bcnts ) );
  } else if ( scnts != 0 && bcnts == 0 && mdl >= scnts / ( ts * scls + tb * sclb ) ) {
    sttstc = ts * mdl - scnts * logf ( ts * mdl ) - scnts * ( 1 - logf ( scnts ) );
  } else if ( scnts != 0 && bcnts == 0 && mdl < scnts / ( ts * scls + tb * sclb ) ) {
    sttstc = ts * ( 1 - scls ) * mdl - tb * sclb * mdl - scnts * logf ( ts * ( 1 - scls ) * mdl + ts * scls * scnts / ( ts * scls + tb * sclb ) ) + scnts * logf ( scnts );
  } else if ( scnts == 0 && bcnts != 0 ) {
    sttstc = ts * mdl - bcnts * logf ( tb * sclb / ( ts * scls + tb * sclb ) );
  } else if ( scnts == 0 && bcnts == 0 ) {
    sttstc = ts * mdl;
  }
  sttstc = 2 * sttstc;
  return sttstc;
}

__host__ __device__ int FindElementIndex ( const float *xx, const int n, const float x ) {
  int ju, jm, jl, jres;
  jl = 0;
  ju = n;
  while ( ju - jl > 1 ) {
    jm = floorf ( 0.5 * ( ju + jl ) );
    if ( x >= xx[jm] ) { jl = jm; } else { ju = jm; }
  }
  jres = jl;
  if ( x == xx[0] ) jres = 0;
  if ( x >= xx[n-1] ) jres = n - 1;
  return jres;
}

__global__ void AssembleArrayOfAbsorptionFactors ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int nmbrOfElmnts, const float *crssctns, const float *abndncs, const int *atmcNmbrs, const float *wlkrs, float *absrptnFctrs ) {
  int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
  int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
  int ttIndx = enIndx + wlIndx * nmbrOfEnrgChnnls;
  int elIndx, effElIndx, crIndx, prIndx;
  float xsctn, clmn, nh;
  if ( enIndx < nmbrOfEnrgChnnls && wlIndx < nmbrOfWlkrs ) {
    if ( NHINDX == NPRS-1 ) {
      elIndx = 0;
      prIndx = elIndx + NHINDX;
      crIndx = elIndx + enIndx * nmbrOfElmnts;
      effElIndx = atmcNmbrs[elIndx] - 1;
      nh = wlkrs[prIndx+wlIndx*NPRS] * 1.E22;
      clmn = abndncs[effElIndx];
      xsctn = clmn * crssctns[crIndx];
      elIndx = 1;
      while ( elIndx < nmbrOfElmnts ) {
        prIndx = elIndx + NHINDX;
        crIndx = elIndx + enIndx * nmbrOfElmnts;
        effElIndx = atmcNmbrs[elIndx] - 1;
        clmn = abndncs[effElIndx]; // * powf ( 10, wlkrs[wlIndx].par[prIndx] );
        xsctn = xsctn + clmn * crssctns[crIndx];
        elIndx += 1;
      }
      absrptnFctrs[ttIndx] = expf ( - nh * xsctn );
    } else if ( NHINDX == NPRS ) {
      absrptnFctrs[ttIndx] = 1;
    }
  }
}

__host__ __device__ float chisquared ( const float d, const float m, const float b, const float scale ) {
  return powf ( d - scale * b - m, 2. ) / d;
}

__host__ __device__ float chi2 ( const float d, const float m ) {
  float s = 0;
  if ( d != 0 ) {
    s = powf ( d - m, 2. ) / d;
  } else if ( m != 0 ) {
    s = powf ( d - m, 2. );
  }
  return s;
}

__global__ void AssembleArrayOfChannelStatistics ( const int nwl, const int nch, const float t_s, const float t_b, const float scal_s, const float scal_b, const float *src, const float *bkg, const float *flx, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nch;
  if ( i < nch && j < nwl ) {
    //stt[t] = PoissonWithBackground ( src[i], bkg[i], flx[t], t_s, t_b, scal_s, scal_b );
    stt[t] = Poisson ( src[i], flx[t], t_s );
    //stt[t] = chisquared ( src[i], flx[t]*t_s, bkg[i], scal_s/scal_b );
    //stt[t] = chi2 ( src[i], flx[t]*t_s );
  }
}

__host__ __device__ float cstat ( const float d, const float m ) {
  float s = 0;
  if ( d > TLR && m > TLR ) {
    s = m - d * logf ( m ) - d * ( 1 - logf ( d ) );
  } else if ( d > TLR && m <= TLR ) {
    s = - d * ( 1 - logf ( d ) );
  } else {
    s = m;
  }
  s *= 2;
  return s;
}

__global__ void arrayOfCStat ( const int nwl, const int nch, const float t, const float *c, const float *f, float *s ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int ij = i + j * nch;
  if ( i < nch && j < nwl ) {
    s[ij] = cstat ( c[i], f[ij] * t );
  }
}

__global__ void arrayOfChiSquareds ( const int nwl, const int nch, const float t, const float *c, const float *f, float *s ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int ij = i + j * nch;
  if ( i < nch && j < nwl ) {
    s[ij] = chi2 ( c[i], f[ij] * t );
  }
}

__global__ void arrayOfChiSquaredsWithBackground ( const int nwl, const int nch, const float t, const float *c, const float *b, const float scale, const float *f, float *s ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int ij = i + j * nch;
  if ( i < nch && j < nwl ) {
    s[ij] = chi2 ( c[i] - scale * b[i], f[ij] * t );
  }
}

__host__ void AssembleArrayOfPhotoelectricCrossections ( const int nmbrOfEnrgChnnls, const int nmbrOfElmnts, int sgFlag, float *enrgChnnls, int *atmcNmbrs, float *crssctns ) {
  int status = 0, versn = sgFlag, indx;
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ ) {
    for ( int j = 0; j < nmbrOfElmnts; j++ ) {
      indx = j + i * nmbrOfElmnts;
      crssctns[indx] = photo_ ( &enrgChnnls[i], &enrgChnnls[i+1], &atmcNmbrs[j], &versn, &status );
    }
  }
}

__global__ void AssembleArrayOfNoticedChannels ( const int nmbrOfChnnls, const float lwrNtcdEnrg, const float hghrNtcdEnrg, const float *lwrChnnlBndrs, const float *hghrChnnlBndrs, const float *gdQltChnnls, float *ntcdChnnls ) {
  int c = threadIdx.x + blockDim.x * blockIdx.x;
  if ( c < nmbrOfChnnls ) {
    ntcdChnnls[c] = ( lwrChnnlBndrs[c] > lwrNtcdEnrg ) * ( hghrChnnlBndrs[c] < hghrNtcdEnrg ) * ( 1 - gdQltChnnls[c] );
  }
}

__host__ int InitializeModel ( Model *mdl ) {
  cudaMallocManaged ( ( void ** ) &mdl[0].atmcNmbrs, ATNMR * sizeof ( int ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].abndncs, ( NELMS + 1 ) * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].RedData, mdl[0].nmbrOfDistBins * mdl[0].numRedCol * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].Dist, mdl[0].nmbrOfDistBins * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].EBV, mdl[0].nmbrOfDistBins * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].RedData1, mdl[0].nmbrOfDistBins1 * mdl[0].numRedCol1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].Dist1, mdl[0].nmbrOfDistBins1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].EBV1, mdl[0].nmbrOfDistBins1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].RedData2, mdl[0].nmbrOfDistBins1 * mdl[0].numRedCol1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].Dist2, mdl[0].nmbrOfDistBins1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].EBV2, mdl[0].nmbrOfDistBins1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].RedData3, mdl[0].nmbrOfDistBins1 * mdl[0].numRedCol1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].Dist3, mdl[0].nmbrOfDistBins1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].EBV3, mdl[0].nmbrOfDistBins1 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].errDist, mdl[0].nmbrOfDistBins * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].errEBV, mdl[0].nmbrOfDistBins * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsaDt, ( mdl[0].numNsaE + 1 ) * ( mdl[0].numNsaT + 1 ) * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsaE, mdl[0].numNsaE * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsaT, mdl[0].numNsaT * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsaFlxs, mdl[0].numNsaE * mdl[0].numNsaT * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsmaxgDt, ( mdl[0].numNsaE + 1 ) * ( mdl[0].numNsaT + 1 ) * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsmaxgE, mdl[0].numNsaE * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsmaxgT, mdl[0].numNsaT * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &mdl[0].nsmaxgFlxs, mdl[0].numNsaE * mdl[0].numNsaT * sizeof ( float ) );
  for ( int i = 0; i < ATNMR; i++ ) { mdl[0].atmcNmbrs[i] = mdl[0].atNm[i]; }
  simpleReadDataFloat ( mdl[0].abndncsFl, mdl[0].abndncs );
  SimpleReadReddenningData ( mdl[0].rddnngFl, mdl[0].nmbrOfDistBins, mdl[0].RedData, mdl[0].Dist, mdl[0].EBV, mdl[0].errDist, mdl[0].errEBV );
  SimpleReadReddenningDataNoErrors ( mdl[0].rddnngFl1, mdl[0].nmbrOfDistBins1, mdl[0].RedData1, mdl[0].Dist1, mdl[0].EBV1 );
  SimpleReadReddenningDataNoErrors ( mdl[0].rddnngFl2, mdl[0].nmbrOfDistBins1, mdl[0].RedData2, mdl[0].Dist2, mdl[0].EBV2 );
  SimpleReadReddenningDataNoErrors ( mdl[0].rddnngFl3, mdl[0].nmbrOfDistBins1, mdl[0].RedData3, mdl[0].Dist3, mdl[0].EBV3 );
  SimpleReadNsaTable ( mdl[0].nsaFl, mdl[0].numNsaE, mdl[0].numNsaT, mdl[0].nsaDt, mdl[0].nsaT, mdl[0].nsaE, mdl[0].nsaFlxs );
  SimpleReadNsmaxgTable ( mdl[0].nsmaxgFl, mdl[0].numNsmaxgE, mdl[0].numNsmaxgT, mdl[0].nsmaxgDt, mdl[0].nsmaxgT, mdl[0].nsmaxgE, mdl[0].nsmaxgFlxs );
  return 0;
}

__global__ void arrayOf2DConditions ( const int dim, const int nwl, const float *bn, const float *xx, float *cc ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    cc[t] = ( bn[0+i*2] < xx[t] ) * ( xx[t] < bn[1+i*2] );
  }
}

__global__ void arrayOfPriors ( const int dim, const int nwl, const float *cn, const float *xx, float *pr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.;
  if ( i < nwl ) {
    pr[i] = ( cn[i] == dim ) * sum + ( cn[i] < dim ) * INF;
  }
}

__global__ void arrayOfPriors1 ( const int dim, const int nwl, const float *cn, const float *nhMd, const float *nhSg, const float *xx, float *pr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float sum; //, theta, kk;
  if ( i < nwl ) {
    //theta = powf ( nhSg[i], 2 ) / nhMd[i];
    //kk = nhMd[i] / theta;
    //sum = ( kk - 1 ) * logf ( xx[NHINDX+i*nwl] ) - xx[NHINDX+i*nwl] / theta;
    sum = 0; //powf ( ( xx[NHINDX+i*nwl] - nhMd[i] ) / nhSg[i], 2 );
    pr[i] = ( cn[i] == dim ) * sum + ( cn[i] < dim ) * INF;
  }
}

__host__ void SimpleReadNsaTable ( const char *flNm, const int numEn, const int numTe, float *data, float *Te, float *En, float *fluxes ) {
  FILE *flPntr;
  float value;
  int i = 0;
  flPntr = fopen ( flNm, "r" );
  while ( fscanf ( flPntr, "%e", &value ) == 1 ) {
    data[i] = value;
    i += 1;
  }
  for (int j = 0; j < numEn; j++) {
    En[j] = log10f ( data[(j+1)*(numTe+1)] );
  }
  for (int j = 0; j < numTe; j++) {
    Te[j] = data[j+1];
  }
  for (int j = 0; j < numEn; j++) {
    for (int i = 0; i < numTe; i++) {
      fluxes[j+i*numEn] = log10f ( data[(i+1)+(j+1)*(numTe+1)] );
    }
  }
  fclose ( flPntr );
}

__host__ void SimpleReadNsmaxgTable ( const char *flNm, const int numEn, const int numTe, float *data, float *Te, float *En, float *fluxes ) {
  FILE *flPntr;
  float value;
  int i = 0;
  flPntr = fopen ( flNm, "r" );
  while ( fscanf ( flPntr, "%e", &value ) == 1 ) {
    data[i] = value;
    i += 1;
  }
  //numTe = (int*)data[0];
  for (int j = 0; j < numTe; j++) {
    Te[j] = data[1+j];
  }
  //numEn = (int*)data[17];
  for (int j = 0; j < numEn; j++) {
    En[j] = log10f ( data[18+j] );
  }
  for (int i = 0; i < numTe; i++) {
    for (int j = 0; j < numEn; j++) {
      fluxes[j+i*numEn] = log10f ( data[(18+numEn)+j+i*numEn] );
    }
  }
  fclose ( flPntr );
}

__host__ void SimpleReadReddenningData ( const char *flNm, const int numDist, float *data, float *Dist, float *EBV, float *errDist, float *errEBV ) {
  FILE *flPntr;
  float value;
  int i = 0;
  flPntr = fopen ( flNm, "r" );
  while ( fscanf (flPntr, "%e", &value ) == 1 ) {
    data[i] = value;
    i += 1;
  }
  for ( int j = 0; j < numDist; j++ ) {
    Dist[j] = log10f ( data[5*j] * 1.E3 );
    EBV[j] = data[5*j+1];
    errDist[j] = log10f ( data[5*j+2] );
    errEBV[j] = log10f ( data[5*j+3] );
  }
  fclose ( flPntr );
}

__host__ void SimpleReadReddenningDataNoErrors ( const char *flNm, const int numDist, float *data, float *Dist, float *EBV ) {
  FILE *flPntr;
  float value;
  int i = 0;
  flPntr = fopen ( flNm, "r" );
  while ( fscanf (flPntr, "%e", &value ) == 1 ) {
    data[i] = value;
    i += 1;
  }
  for ( int j = 0; j < numDist; j++ ) {
    Dist[j] = log10f ( data[2*j] * 1000. );
    EBV[j] = data[2*j+1];
  }
  fclose ( flPntr );
}

__host__ int printSpec ( const Spectrum *spc ) {
  printf ( " spectra -- "  );
  printf ( "\n" );
  for ( int i = 0; i < NSPCTR; i++ ) {
    printf ( " -- "  );
    printf ( "\n" );
    for ( int j = 0; j < 10; j++ ) {
        printf ( " %.8E ", spc[i].mdlFlxs[j] ) ; //IntegrateNsmax ( spc[i].nsa1Flxs[j], spc[i].nsa1Flxs[j+1], spc[i].enrgChnnls[j], spc[i].enrgChnnls[j+1] ) );
    }
    printf ( "\n" );
    printf ( " -- "  );
    printf ( "\n" );
    for ( int j = 0; j < 10; j++ ) {
        printf ( " %.8E ", spc[i].nsa1Flxs[j] );
    }
    printf ( "\n" );
    printf ( " -- "  );
    printf ( "\n" );
    for ( int j = 0; j < 10; j++ ) {
        printf ( " %.8E ", spc[i].enrgChnnls[j] );
    }
    printf ( "\n" );
    printf ( " spectra -- "  );
    printf ( "\n" );
    for ( int j = 0; j < spc[i].nmbrOfChnnls; j++ ) {
        printf ( " %.8E ", spc[i].flddMdlFlxs[j] );
    }
    printf ( "\n" );
    printf ( " stat -- "  );
    printf ( "\n" );
    for ( int j = 0; j < spc[i].nmbrOfChnnls; j++ ) {
        printf ( " %.8E ", spc[i].chnnlSttstcs[j] );
    }
  }
  return 0;
}


#endif // _STRCTRSANDFNCTNS_CU_
