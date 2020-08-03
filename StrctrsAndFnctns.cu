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

__host__ dim3 grid3D ( const int m, const int n, const int w ) {
  dim3 grid ( ( m + THRD1 - 1 ) / THRD1, ( n + THRD2 - 1 ) / THRD2, ( w + THRD3 - 1 ) / THRD3 );
  return grid;
}

__host__ dim3 block3D () {
  dim3 block ( THRD1, THRD2, THRD3 );
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

__global__ void initializeAtRandom ( const int m, const int n, const int ds, const int nwl, const float dlt, const float *stn, const float x0Rad, const float *x0Ang, const float *x0Cen, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if ( i < m && j < n && k < nwl ) {
    xx[0+i*ds+j*ds*m+k*ds*m*n] = x0Cen[0+j*ds+k*ds*n] + x0Rad * cos ( x0Ang[i] ) + stn[0+i*ds+j*ds*m+k*ds*m*n];
    xx[1+i*ds+j*ds*m+k*ds*m*n] = x0Cen[1+j*ds+k*ds*n] + x0Rad * sin ( x0Ang[i] ) + stn[1+i*ds+j*ds*m+k*ds*m*n];
    //chn[0].x0[0+i*chn[0].ds+j*chn[0].ds*chn[0].em] = chn[0].x0Cen[0+j*chn[0].ds]+chn[0].x0Rad*cos(chn[0].x0Ang[i]);
    //chn[0].x0[1+i*chn[0].ds+j*chn[0].ds*chn[0].em] = chn[0].x0Cen[1+j*chn[0].ds]+chn[0].x0Rad*sin(chn[0].x0Ang[i]);
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

__host__ __device__ double funcVV ( const float x ) {
  return pow ( 1 - pow ( x, 2. ), 2. );
}

__global__ void returnXXStatistic ( const int dim, const int nwl, const float *xx, float *s ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  int t1 = i + j * ( dim - 1 );
  float d = dim * 1.;
  if ( i < dim - 1 && j < nwl ) {
    s[t1] = d * pow ( xx[t+1] - xx[t], 2. ) + ( funcVV ( xx[t+1] ) + funcVV ( xx[t] ) ) / d;
  }
}

__global__ void kineticXXStatistic ( const int m, const int n, const int ds, const int nwl, const float *pp, const float *xx, float *ss ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  //float d = m;
  float sum;
  if ( i < m && j < n && k < nwl ) {
    if ( i == m-1 ) {
      sum = 0;
      for ( int l = 0; l < ds; l++ ) {
        sum += pow ( xx[l+j*ds*m+k*ds*m*n] + pp[l+j*ds*m+k*ds*m*n] - xx[l+i*ds+j*ds*m+k*ds*m*n] - pp[l+i*ds+j*ds*m+k*ds*m*n], 2. );
      }
      ss[i+j*m+k*n*m] = sum;
    } else {
      sum = 0;
      for ( int l = 0; l < ds; l++ ) {
        sum += pow ( xx[l+(i+1)*ds+j*ds*m+k*ds*m*n] + pp[l+(i+1)*ds+j*ds*m+k*ds*m*n] - xx[l+i*ds+j*ds*m+k*ds*m*n] - pp[l+i*ds+j*ds*m+k*ds*m*n], 2. );
      }
      ss[i+j*m+k*m*n] = sum;
    }
  }
}

__global__ void distancesXXStatistic ( const int m, const int n, const int ds, const int nwl, const int *t, const int *g, const float *xx, float *ss ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  int nn = n*(n-1)/2;
  float sum;
  if ( i < m && j < nn && k < nwl ) {
    sum = 0;
    for ( int l = 0; l < ds; l++ ) {
      sum += pow ( xx[l+i*ds+t[j]*ds*m+k*ds*m*n] - xx[l+i*ds+g[j]*ds*m+k*ds*m*n], 2. );
    }
    ss[i+j*m+k*m*nn] = sum;
  }
}

__global__ void potentialXXStatistic ( const int m, const int nn, const int nwl, const float *dd, float *uu ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if ( i < m && j < nn && k < nwl ) {
    uu[i+j*m+k*m*nn] = potentialEnergy ( dd[i+j*m+k*m*nn], 10000., 10., 0.01 );
  }
}

__host__ __device__ double potentialEnergy ( const float x, const float b, const float g, const float a ) {
  return b / ( 1 + exp ( g * ( x - a ) ) );
}

__global__ void periodicConditions ( const int m, const int n, const int ds, const int nwl, const float lbox, const float *bound, float *pp, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  float coord, pindx;
  if ( i < m && j < n && k < nwl ) {
    for ( int l = 0; l < ds; l++ ) {
      pindx = pp[l+i*ds+j*ds*m+k*ds*m*n];
      coord = xx[l+i*ds+j*ds*m+k*ds*m*n];
      xx[l+i*ds+j*ds*m+k*ds*m*n] = coord + ( coord < bound[0] ) * lbox - ( coord > bound[1] ) * lbox;
      pp[l+i*ds+j*ds*m+k*ds*m*n] = pindx - ( coord < bound[0] ) * lbox + ( coord > bound[1] ) * lbox;
    }
  }
}

__global__ void arrayOf2DConditions ( const int dim, const int nwl, const float *xx, float *cc ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    cc[t] = ( xx[t] <= 0. );
  }
}

__global__ void setWalkersAtLast ( const int dim, const int nwl, const float *lst, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx[t] = lst[i+j*(dim+1)];
  }
}

__global__ void setStatisticAtLast ( const int dim, const int nwl, const float *lst, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stt[i] = lst[dim+i*(dim+1)];
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

__global__ void chainFunctionU ( const int dim, const int nwl, const int nst, const float *xx, float *ff ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  int t1 = i + j * ( dim - 1 );
  float d = dim * 1.;
  if ( i < dim - 1 && j < nwl*nst ) {
    ff[t1] = 0.5 * ( xx[t+1] + xx[t] ) / d;
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

__global__ void add1DArray ( const int nwl, const float *xx1, const float *xx2, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    xx[i] = 1000. * xx1[i] + xx2[i];
  }
}

__global__ void returnQ ( const int dim, const int n, const float *cnd, const float *s1, const float *s0, const float *zr, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    q[i] = exp ( - 0.5 * ( s1[i] - s0[i] ) ) * powf ( zr[i], dim - 1 );
  }
}

__global__ void returnQM ( const int dim, const int n, const float *s1, const float *s0, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    q[i] = expf ( - 0.5 * ( s1[i] - s0[i] ) );
  }
}

__global__ void updateWalkers ( const int dim, const int nwl, const float *pp1, const float *xx1, const float *q, const float *r, float *pp0, float *xx0 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx0[t] = ( q[j] > r[j] ) * xx1[t] + ( q[j] <= r[j] ) * xx0[t];
    pp0[t] += ( q[j] > r[j] ) * pp1[t];
  }
}

__global__ void updateStatistic ( const int nwl, const float *stt1, const float *q, const float *r, float *stt0 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stt0[i] = ( q[i] > r[i] ) * stt1[i] + ( q[i] <= r[i] ) * stt0[i];
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

__global__ void mapRandomNumbers ( const int nwl, const int ist, const int isb, const float *r, float *zr, int *kr, float *ru ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int rr;
  if ( i < nwl ) {
    rr = i + 0 * nwl + isb * 3 * nwl + ist * 3 * 2 * nwl;
    zr[i] = 1. / ACONST * powf ( r[rr] * ( ACONST - 1 ) + 1, 2. );
    rr = i + 1 * nwl + isb * 3 * nwl + ist * 3 * 2 * nwl;
    kr[i] = ( int ) truncf ( r[rr] * ( nwl - 1 + 0.999999 ) );
    rr = i + 2 * nwl + isb * 3 * nwl + ist * 3 * 2 * nwl;
    ru[i] = r[rr];
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
  cudaMallocManaged ( ( void ** ) &chn[0].ennconst, chn[0].ds * chn[0].em * chn[0].enn * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].kk, chn[0].em * chn[0].en * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].dd, chn[0].em * chn[0].enn * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].uu, chn[0].em * chn[0].enn * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].pot, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].kin, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].bound, 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].x0Cen, chn[0].nwl * chn[0].en * chn[0].ds * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].x0Ang, chn[0].em * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].gindx, chn[0].en * ( chn[0].en - 1 ) / 2 * sizeof ( int ) );
  cudaMallocManaged ( ( void ** ) &chn[0].tindx, chn[0].en * ( chn[0].en - 1 ) / 2 * sizeof ( int ) );
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
  cudaMallocManaged ( ( void ** ) &chn[0].pp, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].pp1, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].pp0, chn[0].dim * chn[0].nwl * sizeof ( float ) );
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
  cudaMallocManaged ( ( void ** ) &chn[0].xxCP, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].zr, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].zuni, chn[0].nst * 2 * chn[0].nwl / 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].kr, chn[0].nwl / 2 * sizeof ( int ) );
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
  cudaMallocManaged ( ( void ** ) &chn[0].cnd, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ccnd, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ff, chn[0].dim * chn[0].nwl * chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].fconst, chn[0].nwl * chn[0].nst * sizeof ( float ) );
  return 0;
}

__host__ int initializeChain ( Cupar *cdp, Chain *chn ) {
  constantArray <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, 1., chn[0].wcnst );
  constantArray <<< grid1D ( chn[0].dim ), THRDSPERBLCK >>> ( chn[0].dim, 1., chn[0].dcnst );
  constantArray <<< grid1D ( chn[0].em*chn[0].enn*chn[0].ds ), THRDSPERBLCK >>> ( chn[0].em*chn[0].enn*chn[0].ds, 1., chn[0].ennconst );
  int tt = 0;
  int gg = 1;
  int ii = 0;
  while ( ii < chn[0].enn ) {
    chn[0].tindx[ii] = tt;
    chn[0].gindx[ii] = gg;
    if ( gg < chn[0].en-1 ) {
      gg += 1;
    } else {
      tt += 1;
      gg = tt + 1;
    }
    ii += 1;
  }
  chn[0].x0Rad = 0.1;
  chn[0].lbox = 1.;
  chn[0].bound[0] = 0.;
  chn[0].bound[1] = 1.;
  for ( int i = 0; i < chn[0].em; i++ ) {
    chn[0].x0Ang[i] = 2. * PI / chn[0].em * i;
  }
  if ( chn[0].indx == 0 ) {
    curandGenerateNormal ( cdp[0].curandGnrtr, chn[0].stn, chn[0].dim * chn[0].nwl, 0, chn[0].dlt );
    curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].x0Cen, chn[0].ds * chn[0].en * chn[0].nwl );
    initializeAtRandom <<< grid3D ( chn[0].em, chn[0].en, chn[0].nwl ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl, chn[0].dlt, chn[0].stn, chn[0].x0Rad, chn[0].x0Ang, chn[0].x0Cen, chn[0].xx );
    constantArray <<< grid1D ( chn[0].dim * chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].dim * chn[0].nwl, 0., chn[0].pp );
    periodicConditions <<< grid3D ( chn[0].em, chn[0].en, chn[0].nwl ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl, chn[0].lbox, chn[0].bound, chn[0].pp, chn[0].xx );
    statistic0 ( cdp, chn );
  } else {
    readLastFromFile ( chn[0].name, chn[0].indx-1, chn[0].dim, chn[0].nwl, chn[0].lst );
    setWalkersAtLast <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].xx );
    setStatisticAtLast <<< grid1D ( chn[0].nwl ), THRDSPERBLCK  >>> ( chn[0].dim, chn[0].nwl, chn[0].lst, chn[0].stt );
  }
  return 0;
}

__host__ int initializeRandomForWalk ( Cupar *cdp, Chain *chn ) {
  curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].uni, chn[0].nst * 2 * chn[0].nwl / 2 );
  curandGenerateNormal ( cdp[0].curandGnrtr, chn[0].stn, chn[0].nst * 2 * chn[0].nwl / 2 * chn[0].dim, 0, 1 );
  return 0;
}

__host__ int initializeRandomForStreach ( Cupar *cdp, Chain *chn ) {
  int n = chn[0].nst * 2 * 3 * chn[0].nwl / 2;
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
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].pp, chn[0].pp0 );
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].pp, chn[0].pp1 );
  sliceArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxXC, chn[0].xx, chn[0].xxC );
  sliceArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt, chn[0].stt0 );
  //constantArray <<< grid1D ( chn[0].dim * chn[0].nwl / 2. ), THRDSPERBLCK >>> ( chn[0].dim * chn[0].nwl / 2., 0., chn[0].pp1 );
  //sliceArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].zuni, chn[0].zr );
  //sliceIntArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].kuni, chn[0].kr );
  mapRandomNumbers <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].ist, chn[0].isb, chn[0].uni, chn[0].zr, chn[0].kr, chn[0].ru );
  //sliceArray <<< grid1D ( nru ), THRDSPERBLCK >>> ( nru, indxRu, chn[0].runi, chn[0].ru );
  permuteWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].kr, chn[0].xxC, chn[0].xxCP );
  substractWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx0, chn[0].xxCP, chn[0].xxCM );
  scale2DArray <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].zr, chn[0].xxCM, chn[0].xxW );
  addWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xxCP, chn[0].xxW, chn[0].xx1 );
  periodicConditions <<< grid3D ( chn[0].em, chn[0].en, chn[0].nwl/2 ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl/2, chn[0].lbox, chn[0].bound, chn[0].pp1, chn[0].xx1 );
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
  //returnXXStatistic <<< grid2D ( chn[0].dim-1, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim-1, chn[0].nwl/2, chn[0].xx1, chn[0].sstt1 );
  kineticXXStatistic <<< grid3D ( chn[0].em, chn[0].en, chn[0].nwl/2 ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl/2, chn[0].pp1, chn[0].xx1, chn[0].kk );
  distancesXXStatistic <<< grid3D ( chn[0].em, chn[0].enn, chn[0].nwl/2 ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl/2, chn[0].tindx, chn[0].gindx, chn[0].xx1, chn[0].dd );
  potentialXXStatistic <<< grid3D ( chn[0].em, chn[0].enn, chn[0].nwl/2 ), block3D () >>> ( chn[0].em, chn[0].enn, chn[0].nwl/2, chn[0].dd, chn[0].uu );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].em*chn[0].en, chn[0].nwl/2, &alpha, chn[0].kk, chn[0].em*chn[0].en, chn[0].ennconst, incxx, &beta, chn[0].kin, incyy );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].em*chn[0].enn, chn[0].nwl/2, &alpha, chn[0].uu, chn[0].em*chn[0].enn, chn[0].ennconst, incxx, &beta, chn[0].pot, incyy );
  add1DArray <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].kin, chn[0].pot, chn[0].stt1 );
  //cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim-1, chn[0].nwl/2, &alpha, chn[0].sstt1, chn[0].dim-1, chn[0].dcnst, incxx, &beta, chn[0].stt1, incyy );
  return 0;
}

__host__ int statistic0 ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  //returnXXStatistic <<< grid2D ( chn[0].dim-1, chn[0].nwl ), block2D () >>> ( chn[0].dim-1, chn[0].nwl, chn[0].xx, chn[0].sstt );
  kineticXXStatistic <<< grid3D ( chn[0].em, chn[0].en, chn[0].nwl ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl, chn[0].pp, chn[0].xx, chn[0].kk );
  distancesXXStatistic <<< grid3D ( chn[0].em, chn[0].enn, chn[0].nwl ), block3D () >>> ( chn[0].em, chn[0].en, chn[0].ds, chn[0].nwl, chn[0].tindx, chn[0].gindx, chn[0].xx, chn[0].dd );
  potentialXXStatistic <<< grid3D ( chn[0].em, chn[0].enn, chn[0].nwl ), block3D () >>> ( chn[0].em, chn[0].enn, chn[0].nwl, chn[0].dd, chn[0].uu );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].em*chn[0].en, chn[0].nwl, &alpha, chn[0].kk, chn[0].em*chn[0].en, chn[0].dcnst, incxx, &beta, chn[0].kin, incyy );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].em*chn[0].enn, chn[0].nwl, &alpha, chn[0].uu, chn[0].em*chn[0].enn, chn[0].dcnst, incxx, &beta, chn[0].pot, incyy );
  add1DArray <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].kin, chn[0].pot, chn[0].stt );
  //cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim-1, chn[0].nwl, &alpha, chn[0].sstt, chn[0].dim-1, chn[0].dcnst, incxx, &beta, chn[0].stt, incyy );
  return 0;
}

__host__ int statisticMetropolis ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  returnStatistic <<< grid2D ( chn[0].dim-1, chn[0].nwl ), block2D () >>> ( chn[0].dim-1, chn[0].nwl, chn[0].xx1, chn[0].sstt1 );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim-1, chn[0].nwl, &alpha, chn[0].sstt1, chn[0].dim-1, chn[0].dcnst, incxx, &beta, chn[0].stt1, incyy );
  return 0;
}

__host__ int walkUpdate ( const Cupar *cdp, Chain *chn ) {
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  returnQM <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].dim, chn[0].nwl/2, chn[0].stt1, chn[0].stt0, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].pp1, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].pp0, chn[0].xx0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt0 );
  insertArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].xx0, chn[0].xx );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt0, chn[0].stt );
  return 0;
}

__host__ int metropolisUpdate ( const Cupar *cdp, Chain *chn ) {
  returnQM <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].dim, chn[0].nwl, chn[0].stt1, chn[0].stt0, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].pp1, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].pp, chn[0].xx );
  updateStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt );
  return 0;
}

__host__ int streachUpdate ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  arrayOf2DConditions <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx1, chn[0].ccnd );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl/2, &alpha, chn[0].ccnd, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].cnd, incyy );
  returnQ <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].dim, chn[0].nwl/2, chn[0].cnd, chn[0].stt1, chn[0].stt0, chn[0].zr, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].pp1, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].pp0, chn[0].xx0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDSPERBLCK >>> ( chn[0].nwl/2, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt0 );
  insertArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].xx0, chn[0].xx );
  insertArray <<< grid1D ( nxx ), THRDSPERBLCK >>> ( nxx, indxX0, chn[0].pp0, chn[0].pp );
  insertArray <<< grid1D ( nss ), THRDSPERBLCK >>> ( nss, indxS0, chn[0].stt0, chn[0].stt );
  return 0;
}

__host__ int saveCurrent ( Chain *chn ) {
  saveWalkers <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].ist, chn[0].xx, chn[0].smpls );
  saveStatistic <<< grid1D ( chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nwl, chn[0].ist, chn[0].stt, chn[0].stat );
  return 0;
}

__host__ int allocateChainForAutoCorr ( Chain *chn ) {
  cudaMallocManaged ( ( void ** ) &chn[0].stps, chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].smOfChn, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cntrlChnFnctn, chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].chnFnctn, chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ftOfChn, chn[0].nst * chn[0].nwl * sizeof ( cufftComplex ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cmSmMtrx, chn[0].nst * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].atcrrFnctn, chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cmSmAtCrrFnctn, chn[0].nst * sizeof ( float ) );
  return 0;
}

__host__ int freeChainForAutoCorr ( const Chain *chn ) {
  cudaFree ( chn[0].stps );
  cudaFree ( chn[0].smOfChn );
  cudaFree ( chn[0].cntrlChnFnctn );
  cudaFree ( chn[0].ftOfChn );
  cudaFree ( chn[0].cmSmMtrx );
  cudaFree ( chn[0].chnFnctn );
  cudaFree ( chn[0].atcrrFnctn );
  cudaFree ( chn[0].cmSmAtCrrFnctn );
  return 0;
}

__host__ int atcrrltnfnctn ( Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int NN[RANK] = { chn[0].nst };
  cufftPlanMany ( &cdp[0].cufftPlan, RANK, NN, NULL, 1, chn[0].nst, NULL, 1, chn[0].nst, CUFFT_C2C, chn[0].nwl );
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
  normArray <<< grid1D ( chn[0].nst ), THRDSPERBLCK >>> ( chn[0].nst, chn[0].atcrrFnctn );
  cudaDeviceSynchronize ();
  cumulativeSumOfAutocorrelationFunction ( chn[0].nst, chn[0].atcrrFnctn, chn[0].cmSmAtCrrFnctn );
  int MM = chooseWindow ( chn[0].nst, 5e0f, chn[0].cmSmAtCrrFnctn );
  chn[0].mmm = MM;
  chn[0].atcTime = 2 * chn[0].cmSmAtCrrFnctn[MM] - 1e0f;
  return 0;
}

__host__ int averagedAutocorrelationFunction ( Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int NN[RANK] = { chn[0].nst };
  cufftPlanMany ( &cdp[0].cufftPlan, RANK, NN, NULL, 1, chn[0].nst, NULL, 1, chn[0].nst, CUFFT_C2C, chn[0].nwl );
  //chainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].nst, 0, chn[0].smpls, chn[0].chnFnctn );
  chainFunctionU <<< grid2D ( chn[0].dim-1, chn[0].nwl * chn[0].nst ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].nst, chn[0].smpls, chn[0].ff );
  constantArray <<< grid1D ( chn[0].nst * chn[0].nwl ), THRDSPERBLCK >>> ( chn[0].nst*chn[0].nwl, alpha, chn[0].fconst );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim-1, chn[0].nwl*chn[0].nst, &alpha, chn[0].ff, chn[0].dim-1, chn[0].fconst, incxx, &beta, chn[0].chnFnctn, incyy );
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
    if ( i >= n - ( dim + 1 ) * nwl ) {
      j = i - ( n - ( dim + 1 ) * nwl );
      lst[j] = value;
    }
    i += 1;
  }
  fclose ( fptr );
}

__host__ void writeChainToFile ( const char *name, const int indx, const int dim, const int nwl, const int nst, const float *smpls, const float *stat ) {
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
      fprintf ( flPntr, " %.8E\n", stat[wlkrIndx+stpIndx*nwl] );
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
  cudaFree ( chn[0].ccnd );
  cudaFree ( chn[0].cnd );
  cudaFree ( chn[0].ff );
  cudaFree ( chn[0].fconst );
  cudaFree ( chn[0].gindx );
  cudaFree ( chn[0].tindx );
  cudaFree ( chn[0].x0Cen );
  cudaFree ( chn[0].x0Ang );
  cudaFree ( chn[0].bound );
  cudaFree ( chn[0].kk );
  cudaFree ( chn[0].dd );
  cudaFree ( chn[0].uu );
  cudaFree ( chn[0].kin );
  cudaFree ( chn[0].pot );
  cudaFree ( chn[0].ennconst );
  cudaFree ( chn[0].pp );
  cudaFree ( chn[0].pp1 );
  cudaFree ( chn[0].pp0 );
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

#endif // _STRCTRSANDFNCTNS_CU_
