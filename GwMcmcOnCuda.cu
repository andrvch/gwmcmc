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

int main ( int argc, char *argv[] ) {
  const int vrb = 1;

  Cupar cdp[1];
  cdp[0].dev = atoi ( argv[1] );

  initializeCuda ( cdp );

  if ( vrb ) {
    printf ( "\n" );
    printf ( ".................................................................\n" );
    printf ( " CUDA device ID: %d\n", cdp[0].dev );
    printf ( " CUDA device Name: %s\n", cdp[0].prop.name );
    printf ( " Driver API: v%d \n", cdp[0].driverVersion[0] );
    printf ( " Runtime API: v%d \n", cdp[0].runtimeVersion[0] );
  }

  Chain chn[1];
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
  const char *spcLst[NSPCTR11] = { spcFl1, spcFl2, spcFl3, spcFl4, spcFl5, spcFl6, spcFl7, spcFl8, spcFl9, spcFl10, spcFl11, spcFl12 };

  chn[0].name = argv[NSPCTR11+2];
  chn[0].nwl = atoi ( argv[NSPCTR11+3] );
  chn[0].nst = atoi ( argv[NSPCTR11+4] );
  chn[0].indx = atoi ( argv[NSPCTR11+5] );
  chn[0].dim = NPRS;
  chn[0].dim1 = chn[0].dim + 3;
  chn[0].dlt = 1.E-4;
  chn[0].nkb = 100;

  Model mdl[1];
  Spectrum spc[NSPCTR];

  const float lwrNtcdEnrg1 = ( float ) atof ( argv[NSPCTR11+6] );
  const float hghrNtcdEnrg1 = ( float ) atof ( argv[NSPCTR11+7] );

  for ( int i = 0; i < NSPCTR; i++ ) {
    spc[i].lwrNtcdEnrg = lwrNtcdEnrg1;
    spc[i].hghrNtcdEnrg = hghrNtcdEnrg1;
  }

  InitializeModel ( mdl );

  SpecInfo ( spcLst, vrb, spc );
  SpecAlloc ( chn, spc );
  SpecData ( cdp, vrb, mdl, spc );

  allocateChain ( chn );

  chn[0].x0[0] = 5.9;
  chn[0].xbnd[0] = 5.5;
  chn[0].xbnd[1] = 6.5;

  chn[0].x0[1] = 0.0;
  chn[0].xbnd[2] = -25.;
  chn[0].xbnd[3] = 25.;

  chn[0].x0[2] = 1.5;
  chn[0].xbnd[4] = -25.;
  chn[0].xbnd[5] = 25.;

  chn[0].x0[3] = -5.;
  chn[0].xbnd[6] = -25.;
  chn[0].xbnd[7] = 25.;

  chn[0].x0[4] = 1.5;
  chn[0].xbnd[8] = -25.;
  chn[0].xbnd[9] = 2.;

  chn[0].x0[5] = -5.;
  chn[0].xbnd[10] = -25.;
  chn[0].xbnd[11] = 25.;

  chn[0].x0[6] = 0.1;
  chn[0].xbnd[12] = 0.;
  chn[0].xbnd[13] = 25.;
/*
  chn[0].x0[7] = -5.;
  chn[0].xbnd[14] = -25.;
  chn[0].xbnd[15] = 25.;

  chn[0].x0[8] = 1.5;
  chn[0].xbnd[16] = -25.;
  chn[0].xbnd[17] = 25.;

  chn[0].x0[9] = -5.;
  chn[0].xbnd[18] = -25.;
  chn[0].xbnd[19] = 25.;

  chn[0].x0[10] = 0.1;
  chn[0].xbnd[20] = 0.;
  chn[0].xbnd[21] = 25.;

  chn[0].x0[11] = -5.;
  chn[0].xbnd[22] = -25.;
  chn[0].xbnd[23] = 25.;

  chn[0].x0[12] = 0.2;
  chn[0].xbnd[24] = 0.;
  chn[0].xbnd[25] = 25.;*/

  initializeChain ( cdp, chn, mdl, spc );

  /*
  cudaDeviceSynchronize ();

  for ( int i = 0; i < spc[0].nmbrOfNtcdChnnls; i++ ) {
    printf ( " %2.2f ", spc[0].flddMdlFlxs[i]  );
    printf ( " %2.2f ", spc[0].chnnlSttstcs[i]  );
  }
  printf ( "\n" );

  for ( int i = 0; i < chn[0].nwl; i++ ) {
    printf ( " %2.2f ", chn[0].stt[i]  );
  }
  printf ( "\n" );
  */

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Start ...                                                  \n" );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  initializeRandomForStreach ( cdp, chn );
  chn[0].ist = 0;
  while ( chn[0].ist < chn[0].nst ) {
    chn[0].isb = 0;
    while ( chn[0].isb < 2 ) {
      streachMove ( cdp, chn );
      modelStatistic1 ( cdp, mdl, chn, spc );
      streachUpdate ( cdp, chn, mdl );
      chn[0].isb += 1;
    }
    saveCurrent ( chn );
    chn[0].ist += 1;
  }

  chainMomentsAndKde ( cdp, chn );

  if ( vrb ) {
    printf ( " Done!\n" );
  }

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Time to generate -- %3.1f ms\n", chn[0].time );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  averagedAutocorrelationFunction ( cdp, chn );

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Autocorrelation time window             -- %i\n", chn[0].mmm );
    printf ( " Autocorrelation time                    -- %.8E\n", chn[0].atcTime );
    printf ( " Autocorrelation time threshold          -- %.8E\n", chn[0].nst / 5e1f );
    printf ( " Effective number of independent samples -- %.8E\n", chn[0].nwl * chn[0].nst / chn[0].atcTime );
    printf ( " Time to compute acor time               -- %3.1f ms\n", chn[0].time );
  }

  sortQKde ( chn );

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Medium                    -- " );
    for ( int i = 0; i < chn[0].dim1; i++ ) {
      printf ( " %2.2f ", chn[0].msmp[i] );
    }
    printf ( "\n" );
    printf ( " Std. deviation            -- " );
    for ( int i = 0; i < chn[0].dim1; i++ ) {
      printf ( " %2.2f ", chn[0].stdsmp[i] );
    }
    printf ( "\n" );
    printf ( " Max pdf (best-fit) values -- " );
    for ( int j = 0; j < chn[0].dim1; j++ ) {
        printf ( " %2.2f ", chn[0].skbin[j+0*chn[0].dim1] );
    }
    printf ( "\n" );
  }

  for ( int i = 0; i < chn[0].dim; i++ ) {
    chn[0].xx[i] = chn[0].skbin[i];
    printf ( " %2.2f ", chn[0].xx[i] );
  }
  printf ( "\n" );

  chn[0].didi[0] = chn[0].skbin[chn[0].dim];

  int incxx = INCXX, incyy = INCYY;
  float alpha = ALPHA, beta = BETA, beta1 = 1.;

  for ( int i = 0; i < NSPCTR; i++ ) {
    AssembleArrayOfAbsorptionFactors <<< grid2D ( spc[i].nmbrOfEnrgChnnls, 1 ), block2D () >>> ( 1, spc[i].nmbrOfEnrgChnnls, ATNMR, spc[i].crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].xx, spc[i].absrptnFctrs );
    BilinearInterpolationNsmax <<< grid2D ( spc[i].nmbrOfEnrgChnnls+1, 1 ), block2D () >>> ( 1, spc[i].nmbrOfEnrgChnnls+1, 0, GRINDX, mdl[0].nsmaxgFlxs, mdl[0].nsmaxgE, mdl[0].nsmaxgT, mdl[0].numNsmaxgE, mdl[0].numNsmaxgT, spc[i].enrgChnnls, chn[0].xx, spc[i].nsa1Flxs );
    AssembleArrayOfModelFluxes2 <<< grid2D ( spc[i].nmbrOfEnrgChnnls, 1 ), block2D () >>> ( i, 1, spc[i].nmbrOfEnrgChnnls, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].enrgChnnls, spc[i].arfFctrs, spc[i].absrptnFctrs, chn[0].xx, spc[i].nsa1Flxs, spc[i].mdlFlxs, chn[0].didi );
    cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[i].nmbrOfNtcdBns, 1, spc[i].nmbrOfEnrgChnnls, spc[i].nmbrOfiVls, &alpha, cdp[0].MatDescr, spc[i].iVls, spc[i].iPntr, spc[i].iIndx, spc[i].mdlFlxs, spc[i].nmbrOfEnrgChnnls, &beta, spc[i].flddMdlFlxs, spc[i].nmbrOfNtcdBns );
    arrayOfWStat <<< grid2D ( spc[i].nmbrOfNtcdBns, 1 ), block2D () >>> ( 1, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].bckgrndExptm, spc[i].backscal_src, spc[i].backscal_bkg, spc[i].srcGrp, spc[i].bkgGrp, spc[i].flddMdlFlxs, spc[i].chnnlSttstcs );
    cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfNtcdBns, 1, &alpha, spc[i].chnnlSttstcs, spc[i].nmbrOfNtcdBns, spc[i].grpVls, INCXX, &beta, spc[i].stat, INCYY );
    arrayOfChiSquaredsWithBackground <<< grid2D ( spc[i].nmbrOfNtcdBns, 1 ), block2D () >>> ( 1, spc[i].nmbrOfNtcdBns, spc[i].srcExptm, spc[i].srcGrp, spc[i].bkgGrp, spc[i].backscal_src/spc[i].backscal_bkg, spc[i].flddMdlFlxs, spc[i].chiSttstcs );
    cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[i].nmbrOfNtcdBns, 1, &alpha, spc[i].chiSttstcs, spc[i].nmbrOfNtcdBns, spc[i].grpVls, INCXX, &beta, spc[i].chi, INCYY );
  }

  cudaDeviceSynchronize ();

  /* Write results to a file */
  simpleWriteDataFloat ( "Autocor.out", chn[0].nst, chn[0].atcrrFnctn );
  simpleWriteDataFloat ( "AutocorCM.out", chn[0].nst, chn[0].cmSmAtCrrFnctn );
  writeSpectraToFile ( "Spectra.out", spc );
  writeWhalesToFile ( chn[0].name, chn[0].indx, chn[0].dim1, chn[0].nwl*chn[0].nst, chn[0].whales );
  writeChainToFile ( chn[0].name, chn[0].indx, chn[0].dim, chn[0].nwl, chn[0].nst, chn[0].smpls, chn[0].stat, chn[0].priors, chn[0].dist, chn[0].chiTwo );

  destroyCuda ( cdp );
  freeChain ( chn );
  FreeModel ( mdl );
  FreeSpec ( spc );

  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits

  cdp[0].err = cudaDeviceReset ();
  if ( cdp[0].err != cudaSuccess ) {
    fprintf ( stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString ( cdp[0].err ) );
    exit ( EXIT_FAILURE );
  }

  return 0;
}

#endif // _GWMCMCCUDA_CU_
