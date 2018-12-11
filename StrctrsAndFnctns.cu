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

__host__ int SpecData ( Cupar *cdp, const int verbose, Model *mdl, Spectrum *spc ) {
  float smOfNtcdChnnls = 0;
  for ( int i = 0; i < NSPCTR; i++ ) {
    if ( verbose == 1 ) {
      printf ( ".................................................................\n" );
      printf ( " Spectrum number  -- %i\n", i );
      printf ( " Spectrum table   -- %s\n", spc[i].srcTbl );
      printf ( " ARF table        -- %s\n", spc[i].arfTbl );
      printf ( " RMF table        -- %s\n", spc[i].rmfTbl );
      printf ( " Background table -- %s\n", spc[i].bckgrndTbl );
    }
    ReadFitsData ( verbose, spc[i].srcTbl, spc[i].arfTbl, spc[i].rmfTbl, spc[i].bckgrndTbl, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfChnnls, spc[i].nmbrOfRmfVls, &spc[i].backscal_src, &spc[i].backscal_bkg, spc[i].srcCnts, spc[i].bckgrndCnts, spc[i].arfFctrs, spc[i].rmfVlsInCsc, spc[i].rmfIndxInCsc, spc[i].rmfPntrInCsc, spc[i].gdQltChnnls, spc[i].lwrChnnlBndrs, spc[i].hghrChnnlBndrs, spc[i].enrgChnnls );
    cusparseScsr2csc ( cdp[0].cusparseHandle, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfChnnls, spc[i].nmbrOfRmfVls, spc[i].rmfVlsInCsc, spc[i].rmfPntrInCsc, spc[i].rmfIndxInCsc, spc[i].rmfVls, spc[i].rmfIndx, spc[i].rmfPntr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO );
    AssembleArrayOfNoticedChannels <<< grid1D ( spc[i].nmbrOfChnnls ), THRDS >>> ( spc[i].nmbrOfChnnls, spc[i].lwrNtcdEnrg, spc[i].hghrNtcdEnrg, spc[i].lwrChnnlBndrs, spc[i].hghrChnnlBndrs, spc[i].gdQltChnnls, spc[i].ntcdChnnls );
    cublasSdot ( cdp[0].cublasHandle, spc[i].nmbrOfChnnls, spc[i].ntcdChnnls, INCXX, spc[i].ntcdChnnls, INCYY, &spc[i].smOfNtcdChnnls );
    cudaDeviceSynchronize ( );
    smOfNtcdChnnls = smOfNtcdChnnls + spc[i].smOfNtcdChnnls;
    AssembleArrayOfPhotoelectricCrossections ( spc[i].nmbrOfEnrgChnnls, ATNMR, mdl[0].sgFlg, spc[i].enrgChnnls, mdl[0].atmcNmbrs, spc[i].crssctns );
    if ( verbose == 1 ) {
      printf ( " Number of energy channels                = %i\n", spc[i].nmbrOfEnrgChnnls );
      printf ( " Number of instrument channels            = %i\n", spc[i].nmbrOfChnnls );
      printf ( " Number of nonzero elements of RMF matrix = %i\n", spc[i].nmbrOfRmfVls );
      printf ( " Exposure time                            = %.8E\n", spc[i].srcExptm );
      printf ( " Exposure time (background)               = %.8E\n", spc[i].bckgrndExptm );
      printf ( " Number of used instrument channels -- %4.0f\n", spc[i].smOfNtcdChnnls );
      printf ( " Backscale src -- %4.0f\n", spc[i].backscal_src );
      printf ( " Backscale bkg -- %4.0f\n", spc[i].backscal_bkg );
    }
  }
  if ( verbose == 1 ) {
    printf ( ".................................................................\n" );
    printf ( " Total number of used instrument channels -- %4.0f\n", smOfNtcdChnnls );
    printf ( " Number of degrees of freedom -- %4.0f\n", smOfNtcdChnnls - NPRS );
  }
  return 0;
}

__host__ int SpecInfo ( const char *spcLst[NSPCTR], const int verbose, Spectrum *spc ) {
  for ( int i = 0; i < NSPCTR; i++ ) {
    ReadFitsInfo ( spcLst[i], &spc[i].nmbrOfEnrgChnnls, &spc[i].nmbrOfChnnls, &spc[i].nmbrOfRmfVls, &spc[i].srcExptm, &spc[i].bckgrndExptm, spc[i].srcTbl, spc[i].arfTbl, spc[i].rmfTbl, spc[i].bckgrndTbl );
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
  }
  return 0;
}

__global__ void arrayOf2DConditions ( const int dim, const int nwl, const float *bn, const float *xx, float *cc ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    cc[t] = ( bn[0+i*2] <= xx[t] ) * ( xx[t] < bn[1+i*2] );
  }
}

__global__ void arrayOfPriors ( const int dim, const int nwl, const float *cn, const float *xx, float *pr ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 2. * logf ( 2 * xx[0+i*dim] );
  if ( i < nwl ) {
    pr[i] = ( cn[i] == dim ) * sum + ( cn[i] < dim ) * INF;
  }
}

__host__ __device__ int binNumber ( const int nbn, const float tms, const float fr, const float ph ) {
  float jt, jtFr, jtJt, jtInt;
  int jIndx;
  jt = 1 + nbn * fmodf ( 2 * PI * ( fr * tms + ph ), 2 * PI ) / 2 / PI;
  jtFr = modff( jt, &jtInt );
  jtJt = jt - jtFr;
  jIndx = llroundf ( jtJt );
  return jIndx;
}

__global__ void arrayOfBinTimes ( const int nph, const int nbm, const int nwl, const float *xx, const float *at, float *nn ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  int t = i + (j+k*nbm) * nph;
  if ( i < nph && j < nbm && k < nwl ) {
    nn[t] = ( j + 1 == binNumber ( nbm, at[i], xx[0], xx[1] ) );
  }
}

__global__ void arrayOfMultiplicity ( const int nph, const int nbm, const int nwl, const float scale, const float *nTms, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nbm;
  if ( i < nbm && j < nwl ) {
    stt[t] = -2. * ( scale / nbm + ( nbm - 1 ) * logf ( 2 * PI ) / nbm - 0.5 * logf ( nph ) / nbm + nTms[t] * logf ( nTms[t] / nph ) + 0.5 * logf ( nTms[t] ) );
  }
}

__global__ void arrayOfStat ( const int nbm, const float *mt, float *mstt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nbm - 1 ) {
    mstt[i] = - 2. * mt[1+i];
  }
}

__global__ void saveNumbers ( const int nbm, const int nwl, const int ist, const float *nt, float *numbers ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nbm;
  if ( i < nbm && j < nwl ) {
    numbers[t+ist*nbm*nwl] = nt[t];
  }
}

__global__ void updateNumbers ( const int nbm, const int nwl, const float *nt1, const float *q, const float *r, float *nt0 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nbm;
  if ( i < nbm && j < nwl ) {
    //if ( q[j] > r[j] ) {
    nt0[t] = ( q[j] > r[j] ) * nt1[t] + ( q[j] <= r[j] ) * nt0[t];
    //}
  }
}

__host__ int modelStatistic ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  dim3 block3 ( 1024, 1, 1 );
  dim3 grid3 = grid3D ( chn[0].nph, chn[0].nbm, chn[0].nwl, block3 );
  arrayOfBinTimes <<< grid3, block3 >>> ( chn[0].nph, chn[0].nbm, chn[0].nwl, chn[0].xx, chn[0].atms, chn[0].nnt );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].nph, chn[0].nbm * chn[0].nwl, &alpha, chn[0].nnt, chn[0].nph, chn[0].pcnst, INCXX, &beta, chn[0].nt, INCYY );
  arrayOfMultiplicity <<< grid2D ( chn[0].nbm, chn[0].nwl ), block2D () >>> ( chn[0].nph, chn[0].nbm, chn[0].nwl, chn[0].scale, chn[0].nt, chn[0].mmt );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].nbm, chn[0].nwl, &alpha, chn[0].mmt, chn[0].nbm, chn[0].bcnst, incxx, &beta, chn[0].stt, incyy );
  arrayOf2DConditions <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xbnd, chn[0].xx, chn[0].ccnd );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl, &alpha, chn[0].ccnd, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].cnd, incyy );
  arrayOfPriors  <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].dim, chn[0].nwl, chn[0].cnd, chn[0].xx, chn[0].prr );
  return 0;
}

__host__ int modelStatistic1 ( const Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  dim3 block3 ( 1024, 1, 1 );
  dim3 grid3 = grid3D ( chn[0].nph, chn[0].nbm, chn[0].nwl, block3 );
  arrayOfBinTimes <<< grid3, block3 >>> ( chn[0].nph, chn[0].nbm, chn[0].nwl, chn[0].xx1, chn[0].atms, chn[0].nnt );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].nph, chn[0].nbm * chn[0].nwl, &alpha, chn[0].nnt, chn[0].nph, chn[0].pcnst, INCXX, &beta, chn[0].nt1, INCYY );
  arrayOfMultiplicity <<< grid2D ( chn[0].nbm, chn[0].nwl ), block2D () >>> ( chn[0].nph, chn[0].nbm, chn[0].nwl, chn[0].scale, chn[0].nt1, chn[0].mmt );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].nbm, chn[0].nwl, &alpha, chn[0].mmt, chn[0].nbm, chn[0].bcnst, incxx, &beta, chn[0].stt1, incyy );
  arrayOf2DConditions <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xbnd, chn[0].xx1, chn[0].ccnd );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].dim, chn[0].nwl, &alpha, chn[0].ccnd, chn[0].dim, chn[0].dcnst, incxx, &beta, chn[0].cnd, incyy );
  arrayOfPriors  <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].dim, chn[0].nwl, chn[0].cnd, chn[0].xx1, chn[0].prr1 );
  return 0;
}


__host__ dim3 grid3D ( const int n, const int m, const int l, const dim3 block ) {
  dim3 grid ( ( n + block.x - 1 ) / block.x, ( m + block.y - 1 ) / block.y, ( l + block.z - 1 ) / block.z );
  return grid;
}

__host__ int readTimesInfo ( const char *spcFl, int *nmbrOfPhtns, float *srcExptm ) {
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

__host__ int readTimesData ( const char *spcFl, const int nmbrOfPhtns, float *arrTms ) {
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


__host__ int grid1D ( const int n ) {
  int b = ( n + THRDS - 1 ) / THRDS;
  return b;
}

__host__ dim3 grid2D ( const int n, const int m ) {
  dim3 grid ( ( n + THRDS - 1 ) / THRDS, ( m + THRDS - 1 ) / THRDS );
  return grid;
}

__host__ dim3 block2D () {
  dim3 block ( THRDS, THRDS );
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

__global__ void setWalkersAtLast ( const int dim, const int nwl, const int nbm, const float *lst, float *xx ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx[t] = lst[i+j*(dim+1+nbm)];
  }
}

__global__ void setStatisticAtLast ( const int dim, const int nwl, const int nbm, const float *lst, float *stt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < nwl ) {
    stt[i] = lst[dim+i*(dim+1+nbm)];
  }
}

__global__ void setNumbersAtLast ( const int dim, const int nwl, const int nbm, const float *lst, float *nt ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * nbm;
  if ( i < nbm && j < nwl ) {
    nt[t] = lst[dim+1+i+j*(dim+1+nbm)];
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

__global__ void returnQM ( const int dim, const int n, const float *s1, const float *s0, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    q[i] = expf ( - 0.5 * ( s1[i] - s0[i] ) );
  }
}

__global__ void returnQM1 ( const int dim, const int n, const float *p1, const float *p0, const float *s1, const float *s0, float *q ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    q[i] = expf ( - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) );
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

__global__ void metropolisPoposal3 ( const int dim, const int nwl, const int isb, const float *sigma, const float *xx, const float *rr, float *xx1 ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  if ( i < dim && j < nwl ) {
    xx1[t] = xx[t] + ( i == isb ) * sigma[isb] * rr[j];
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
  cudaMallocManaged ( ( void ** ) &chn[0].prr, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].prr1, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].xbnd, chn[0].dim * 2 * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].cnd, chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].ccnd, chn[0].dim * chn[0].nwl * sizeof ( float ) );
  return 0;
}

__host__ int allocateTimes ( Chain *chn ) {
  cudaMallocManaged ( ( void ** ) &chn[0].atms, chn[0].nph * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].nnt, chn[0].nph * chn[0].nbm * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].bnn, chn[0].nph * chn[0].nwl * sizeof ( int ) );
  cudaMallocManaged ( ( void ** ) &chn[0].nt, chn[0].nbm * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].nt1, chn[0].nbm * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].numbers, chn[0].nbm * chn[0].nwl * chn[0].nst * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].mmt, chn[0].nbm * chn[0].nwl * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].mt, chn[0].nbm * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].mstt, chn[0].nbm * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].bcnst, chn[0].nbm * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].pcnst, chn[0].nph * sizeof ( float ) );
  cudaMallocManaged ( ( void ** ) &chn[0].sigma, chn[0].dim * sizeof ( float ) );
  return 0;
}


__host__ int initializeChain ( Cupar *cdp, Chain *chn ) {
  constantArray <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, 1., chn[0].wcnst );
  constantArray <<< grid1D ( chn[0].dim ), THRDS >>> ( chn[0].dim, 1., chn[0].dcnst );
  constantArray <<< grid1D ( chn[0].nbm ), THRDS >>> ( chn[0].nbm, 1., chn[0].bcnst );
  constantArray <<< grid1D ( chn[0].nph ), THRDS >>> ( chn[0].nph, 1., chn[0].pcnst );
  if ( chn[0].indx == 0 ) {
    curandGenerateNormal ( cdp[0].curandGnrtr, chn[0].stn, chn[0].dim * chn[0].nwl, 0, 1 );
    initializeAtRandom <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].dlt, chn[0].x0, chn[0].stn, chn[0].xx );
    //statistic0 ( cdp, chn );
    modelStatistic ( cdp, chn );
  } else {
    readLastFromFile ( chn[0].name, chn[0].indx-1, chn[0].dim, chn[0].nwl, chn[0].nbm, chn[0].lst );
    setWalkersAtLast <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].nbm, chn[0].lst, chn[0].xx );
    setStatisticAtLast <<< grid1D ( chn[0].nwl ), THRDS  >>> ( chn[0].dim, chn[0].nwl, chn[0].nbm, chn[0].lst, chn[0].stt );
    setNumbersAtLast <<< grid2D ( chn[0].nbm, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].nbm, chn[0].lst, chn[0].nt );
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
  sliceArray <<< grid1D ( nxx ), THRDS >>> ( nxx, indxX0, chn[0].xx, chn[0].xx0 );
  sliceArray <<< grid1D ( nxx ), THRDS >>> ( nxx, indxXC, chn[0].xx, chn[0].xxC );
  sliceArray <<< grid1D ( nss ), THRDS >>> ( nss, indxS0, chn[0].stt, chn[0].stt0 );
  sliceArray <<< grid1D ( nrn ), THRDS >>> ( nrn, indxRn, chn[0].stn, chn[0].zz );
  sliceArray <<< grid1D ( nru ), THRDS >>> ( nru, indxRu, chn[0].uni, chn[0].ru );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_N, chn[0].dim, chn[0].nwl/2, &alpha, chn[0].xxC, chn[0].dim, chn[0].wcnst, incxx, &beta, chn[0].xCM, incyy );
  scaleArray <<< grid1D ( chn[0].dim ), THRDS >>> ( chn[0].dim, 2./chn[0].nwl, chn[0].xCM );
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
  sliceArray <<< grid1D ( nxx ), THRDS >>> ( nxx, indxX0, chn[0].xx, chn[0].xx0 );
  sliceArray <<< grid1D ( nxx ), THRDS >>> ( nxx, indxXC, chn[0].xx, chn[0].xxC );
  sliceArray <<< grid1D ( nss ), THRDS >>> ( nss, indxS0, chn[0].stt, chn[0].stt0 );
  //sliceArray <<< grid1D ( nru ), THRDS >>> ( nru, indxRu, chn[0].zuni, chn[0].zr );
  //sliceIntArray <<< grid1D ( nru ), THRDS >>> ( nru, indxRu, chn[0].kuni, chn[0].kr );
  mapRandomNumbers <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].nwl/2, chn[0].ist, chn[0].isb, chn[0].uni, chn[0].zr, chn[0].kr, chn[0].ru );
  //sliceArray <<< grid1D ( nru ), THRDS >>> ( nru, indxRu, chn[0].runi, chn[0].ru );
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
  sliceArray <<< grid1D ( nrn ), THRDS >>> ( nrn, iRn, chn[0].stn1, chn[0].rr );
  sliceArray <<< grid1D ( nru ), THRDS >>> ( nru, iRu, chn[0].uni, chn[0].ru );
  metropolisPoposal3 <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].isb, chn[0].sigma, chn[0].xx, chn[0].rr, chn[0].xx1 );
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
  returnQM <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].dim, chn[0].nwl/2, chn[0].stt1, chn[0].stt0, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].xx0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].nwl/2, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt0 );
  insertArray <<< grid1D ( nxx ), THRDS >>> ( nxx, indxX0, chn[0].xx0, chn[0].xx );
  insertArray <<< grid1D ( nss ), THRDS >>> ( nss, indxS0, chn[0].stt0, chn[0].stt );
  return 0;
}

__host__ int metropolisUpdate ( const Cupar *cdp, Chain *chn ) {
  returnQM1 <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].dim, chn[0].nwl, chn[0].prr1, chn[0].prr, chn[0].stt1, chn[0].stt, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].xx );
  updateStatistic <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt );
  updateNumbers <<< grid2D ( chn[0].nbm, chn[0].nwl ), block2D () >>> ( chn[0].nbm, chn[0].nwl, chn[0].nt1, chn[0].q, chn[0].ru, chn[0].nt );
  return 0;
}

__host__ int streachUpdate ( const Cupar *cdp, Chain *chn ) {
  int nxx = chn[0].dim * chn[0].nwl / 2;
  int indxX0 = chn[0].isb * nxx;
  int nss = chn[0].nwl / 2;
  int indxS0 = chn[0].isb * nss;
  returnQ <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].dim, chn[0].nwl/2, chn[0].stt1, chn[0].stt0, chn[0].zr, chn[0].q );
  updateWalkers <<< grid2D ( chn[0].dim, chn[0].nwl/2 ), block2D () >>> ( chn[0].dim, chn[0].nwl/2, chn[0].xx1, chn[0].q, chn[0].ru, chn[0].xx0 );
  updateStatistic <<< grid1D ( chn[0].nwl/2 ), THRDS >>> ( chn[0].nwl/2, chn[0].stt1, chn[0].q, chn[0].ru, chn[0].stt0 );
  insertArray <<< grid1D ( nxx ), THRDS >>> ( nxx, indxX0, chn[0].xx0, chn[0].xx );
  insertArray <<< grid1D ( nss ), THRDS >>> ( nss, indxS0, chn[0].stt0, chn[0].stt );
  return 0;
}

__host__ int saveCurrent ( Chain *chn ) {
  saveWalkers <<< grid2D ( chn[0].dim, chn[0].nwl ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].ist, chn[0].xx, chn[0].smpls );
  saveStatistic <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, chn[0].ist, chn[0].stt, chn[0].stat );
  saveNumbers <<< grid2D ( chn[0].nbm, chn[0].nwl ), block2D () >>> ( chn[0].nbm, chn[0].nwl, chn[0].ist, chn[0].nt, chn[0].numbers );
  return 0;
}

__host__ int averagedAutocorrelationFunction ( Cupar *cdp, Chain *chn ) {
  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA;
  int NN[RANK] = { chn[0].nst };
  cufftPlanMany ( &cdp[0].cufftPlan, RANK, NN, NULL, 1, chn[0].nst, NULL, 1, chn[0].nst, CUFFT_C2C, chn[0].nwl );
  chainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].dim, chn[0].nwl, chn[0].nst, 0, chn[0].smpls, chn[0].chnFnctn );
  constantArray <<< grid1D ( chn[0].nst ), THRDS >>> ( chn[0].nst, alpha / chn[0].nst, chn[0].stps );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_N, chn[0].nwl, chn[0].nst, &alpha, chn[0].chnFnctn, chn[0].nwl, chn[0].stps, incxx, &beta, chn[0].smOfChn, incyy );
  shiftWalkers <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].nwl, chn[0].nst, chn[0].chnFnctn, chn[0].smOfChn, chn[0].cntrlChnFnctn );
  testChainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].nwl, chn[0].nst, 0, chn[0].cntrlChnFnctn, chn[0].ftOfChn );
  cufftExecC2C ( cdp[0].cufftPlan, ( cufftComplex * ) chn[0].ftOfChn, ( cufftComplex * ) chn[0].ftOfChn, CUFFT_FORWARD );
  complexPointwiseMultiplyByConjugateAndScale <<< grid2D ( chn[0].nst, chn[0].nwl ), block2D () >>> ( chn[0].nst, chn[0].nwl, alpha / chn[0].nst, chn[0].ftOfChn );
  cufftExecC2C ( cdp[0].cufftPlan, ( cufftComplex * ) chn[0].ftOfChn, ( cufftComplex * ) chn[0].ftOfChn, CUFFT_INVERSE );
  testChainFunction <<< grid2D ( chn[0].nwl, chn[0].nst ), block2D () >>> ( chn[0].nwl, chn[0].nst, 1, chn[0].cntrlChnFnctn, chn[0].ftOfChn );
  constantArray <<< grid1D ( chn[0].nwl ), THRDS >>> ( chn[0].nwl, alpha / chn[0].nwl, chn[0].wcnst );
  cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, chn[0].nwl, chn[0].nst, &alpha, chn[0].cntrlChnFnctn, chn[0].nwl, chn[0].wcnst, incxx, &beta, chn[0].atcrrFnctn, incyy );
  //scaleArray <<< grid1D ( chn[0].nst ), THRDS >>> ( chn[0].nst, 1. / chn[0].atcrrFnctn[0], chn[0].atcrrFnctn );
  normArray <<< grid1D ( chn[0].nst ), THRDS >>> ( chn[0].nst, chn[0].atcrrFnctn );
  cudaDeviceSynchronize ();
  cumulativeSumOfAutocorrelationFunction ( chn[0].nst, chn[0].atcrrFnctn, chn[0].cmSmAtCrrFnctn );
  int MM = chooseWindow ( chn[0].nst, 5e0f, chn[0].cmSmAtCrrFnctn );
  chn[0].mmm = MM;
  chn[0].atcTime = 2 * chn[0].cmSmAtCrrFnctn[MM] - 1e0f;
  return 0;
}

__host__ void readLastFromFile ( const char *name, const int indx, const int dim, const int nwl, const int nbm, float *lst ) {
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
    if ( i >= n - ( dim + 1 + nbm ) * nwl ) {
      j = i - ( n - ( dim + 1 + nbm ) * nwl );
      lst[j] = value;
    }
    i += 1;
  }
  fclose ( fptr );
}

__host__ void writeChainToFile ( const char *name, const int indx, const int dim, const int nwl, const int nst, const int nbm, const float *smpls, const float *stat, const float *numbers ) {
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
      int nnn = prmtrIndx;
      while ( prmtrIndx - nnn < nbm ) {
        fprintf ( flPntr, " %.8E ", numbers[(prmtrIndx-nnn)+wlkrIndx*nbm+stpIndx*nwl*nbm] );
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
  cudaFree ( chn[0].prr );
  cudaFree ( chn[0].prr1 );
  cudaFree ( chn[0].cnd );
  cudaFree ( chn[0].ccnd );
  cudaFree ( chn[0].xbnd );
}

__host__ void freeTimes ( const Chain *chn ) {
  cudaFree ( chn[0].atms );
  cudaFree ( chn[0].nnt );
  cudaFree ( chn[0].nt );
  cudaFree ( chn[0].nt1 );
  cudaFree ( chn[0].numbers );
  cudaFree ( chn[0].mmt );
  cudaFree ( chn[0].mt );
  cudaFree ( chn[0].mstt );
  cudaFree ( chn[0].pcnst );
  cudaFree ( chn[0].bcnst );
  cudaFree ( chn[0].sigma );
  cudaFree ( chn[0].bnn );
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
  printf ( " atms -- "  );
  printf ( "\n" );
  for ( int i = 0; i < 10; i++ ) {
    printf ( " %2.4f ", chn[0].atms[i] );
  }
  printf ( "\n" );
  /*printf ( " nnt -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nbm; i++ ) {
    for ( int j = 0; j < chn[0].nbm; j++ ) {
      for ( int k = 0; k < 10; k++ ) {
        printf ( " %2.4f ", chn[0].nnt[k+(j+i*chn[0].nbm)*chn[0].nph] );
      }
      printf ( "\n" );
    }
  }*/
  printf ( "\n" );
  printf ( " nt -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nbm; i++ ) {
    for ( int j = 0; j < chn[0].nbm; j++ ) {
        printf ( " %2.4f ", chn[0].nt[i+j*chn[0].nbm] );
    }
    printf ( "\n" );
  }
  printf ( "\n" );
  printf ( "\n" );
  printf ( " stt0 -- "  );
  printf ( "\n" );
  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.4f ", chn[0].stt0[i] );
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

__host__ int ReadFitsInfo ( const char *spcFl, int *nmbrOfEnrgChnnls, int *nmbrOfChnnls, int *nmbrOfRmfVls, float *srcExptm, float *bckgrndExptm, char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] )
{
  fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
  int status = 0, intnull = 0, anynull = 0, colnum;
  char card[FLEN_CARD], colNgr[] = "N_GRP", colNch[] = "N_CHAN";
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
  fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
  if ( status == 0 && BACKIN == 1 )
  {
    fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", bckgrndExptm, NULL, &status );
  }
  else
  {
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
  int sum = 0;
  for ( int i = 0; i < *nmbrOfEnrgChnnls; i++ )
  {
    fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
    fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, n_chan_vec, &anynull, &status );
    for ( int j = 0; j < n_grp[i]; j++ )
    {
      sum = sum + n_chan_vec[j];
    }
  }
  *nmbrOfRmfVls = sum;
  free ( n_chan_vec );
  free ( n_grp );
  return 0;
}

__host__ int ReadFitsData ( const int verbose, const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD], const int nmbrOfEnrgChnnls, const int nmbrOfChnnls, const int nmbrOfRmfVls, float *backscal_src, float *backscal_bkg, float *srcCnts, float *bckgrndCnts, float *arfFctrs, float *rmfVlsInCsc, int *rmfIndxInCsc, int *rmfPntrInCsc, float *gdQltChnnls, float *lwrChnnlBndrs, float *hghrChnnlBndrs, float *enrgChnnls )
{
  fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
  int status = 0, anynull, colnum, intnull = 0, rep_chan = 100;
  char card[FLEN_CARD], EboundsTable[FLEN_CARD], Telescop[FLEN_CARD];
  char colNgr[]="N_GRP", colNch[]="N_CHAN",  colFch[]="F_CHAN", colCounts[]="COUNTS", colSpecResp[]="SPECRESP", colEnLo[]="ENERG_LO", colEnHi[]="ENERG_HI", colMat[]="MATRIX", colEmin[]="E_MIN", colEmax[]="E_MAX";
  float floatnull;
  /* Read Spectrum: */
  fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
  fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
  snprintf ( EboundsTable, sizeof ( EboundsTable ), "%s%s", card, "[EBOUNDS]" );
  fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", backscal_src, NULL, &status );
  fits_read_key ( ftsPntr, TSTRING, "TELESCOP", Telescop, NULL, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, srcCnts, &anynull, &status );
  /* Read ARF FILE: */
  fits_open_file ( &ftsPntr, arfTbl, READONLY, &status );
  fits_get_colnum ( ftsPntr, CASEINSEN, colSpecResp, &colnum, &status );
  fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, arfFctrs, &anynull, &status );
  /* Read Background: */
  fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
  if ( status == 0 && BACKIN == 1 )
  {
    fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", backscal_bkg, NULL, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, bckgrndCnts, &anynull, &status );
  }
  else
  {
    if ( verbose == 1)
    {
      printf ( " Warning: Background table is not used, background exposure and background are set to 0.\n " );
    }
    for ( int i = 0; i < nmbrOfChnnls; i++ )
    {
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
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
  {
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
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
  {
    fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
    fits_read_col_int ( ftsPntr, colnum, i+1, 1, n_grp[i], intnull, n_chan_vec, &anynull, &status );
    for ( int j = 0; j < rep_chan; j++ )
    {
      n_chan[i*rep_chan+j] = n_chan_vec[j];
    }
  }
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
  {
    fits_get_colnum ( ftsPntr, CASEINSEN, colFch, &colnum, &status );
    fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, f_chan_vec, &anynull, &status );
    for ( int j = 0; j < rep_chan; j++ )
    {
      f_chan[i*rep_chan+j] = f_chan_vec[j];
    }
  }
  int sum = 0;
  rmfPntrInCsc[0] = 0;
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
  {
    for ( int j = 0; j < n_grp[i]; j++ )
    {
      sum = sum + n_chan[rep_chan*i+j];
    }
    rmfPntrInCsc[i+1] = sum;
  }
  int m = 0;
  if ( nmbrOfChnnls != 1024 )
  {
    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
      for ( int j = 0; j < n_grp[i]; j++ )
      {
        for ( int k = f_chan[rep_chan*i+j] ; k < f_chan[rep_chan*i+j] + n_chan[rep_chan*i+j]; k++ )
        {
          rmfIndxInCsc[m] = k;
          m = m + 1;
        }
      }
    }
  }
  else if ( nmbrOfChnnls == 1024 )
  {
    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
      for ( int j = 0; j < n_grp[i]; j++ )
      {
        for ( int k = f_chan[rep_chan*i+j] - 1; k < f_chan[rep_chan*i+j] - 1 + n_chan[rep_chan*i+j]; k++ )
        {
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
  for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
  {
    sum = rmfPntrInCsc[i+1] - rmfPntrInCsc[i];
    fits_read_col ( ftsPntr, TFLOAT, colnum, i+1, 1, sum, &floatnull, rmf_vec, &anynull, &status );
    for ( int k = 0; k < sum; k++ )
    {
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

#endif // _STRCTRSANDFNCTNS_CU_
