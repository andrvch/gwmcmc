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
  chn[0].dlt = 1.E-4;

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

  /*
  printf ( " Grouping Information -- \n " );
  int count = 0;
  for ( int i = 0; i < spc[0].nmbrOfBns; i++ ) {
    count += spc[0].grpPntr[i+1] - spc[0].grpPntr[i];
    printf ( " %i ", i );
    printf ( " %i ", spc[0].grpPntr[i+1] - spc[0].grpPntr[i] );
    for ( int j = spc[0].grpPntr[i]; j < spc[0].grpPntr[i+1]; j++ ) {
      printf ( " %2.0f ", spc[0].grpng[j] );
    }
    printf ( "\n" );
  }
  printf ( " Number of grouping bins -- %i\n ", count );
  printf ( " final bin -- %i\n", spc[0].grpPntr[spc[0].nmbrOfBns] - spc[0].grpPntr[spc[0].nmbrOfBns-1] );
  for ( int i = 0; i < spc[0].nmbrOfChnnls; i++ ) {
    printf ( " %i ", spc[0].grpIndx[i] );
  }
  printf ( "\n" );
  for ( int i = 0; i < spc[0].nmbrOfChnnls; i++ ) {
    printf ( " %2.0f ", spc[0].grpVls[i] );
  }
  printf ( "\n" );

  printf ( " %i\n", spc[0].nmbrOfNtcdBns );
  printf ( " %i\n", spc[0].nmbrOfUsdBns );
  printf ( " %i\n", spc[0].lwrBn );
  printf ( " %i\n", spc[0].hghrBn );
  printf ( " %i\n", spc[0].nmbrOfgrpIgnVls );

  for ( int i = 0; i < spc[0].nmbrOfgrpIgnVls; i++ ) {
    printf ( " %2.0f ", spc[0].grpIgnVls[i] );
  }
  printf ( " \n " );

  for ( int i = 0; i < spc[0].nmbrOfgrpIgnVls; i++ ) {
    printf ( " %i ", spc[0].grpIgnIndx[i] );
  }
  printf ( " \n " );

  for ( int i = 0; i < spc[0].nmbrOfNtcdBns+1; i++ ) {
    printf ( " %i ", spc[0].grpIgnPntr[i] );
  }
  printf ( " \n " );

  for ( int j = 0; j < NSPCTR; j++ ) {
    for ( int i = 0; i < spc[j].nmbrOfUsdBns; i++ ) {
      printf ( " %2.0f ", spc[j].srcGrp[i] );
    }
    printf ( " \n " );
  }*/

  allocateChain ( chn );

  chn[0].x0[0] = 5.9;
  chn[0].xbnd[0] = 5.5;
  chn[0].xbnd[1] = 6.5;

  chn[0].x0[1] = 0.0;
  chn[0].xbnd[2] = -5.;
  chn[0].xbnd[3] = 5;

  chn[0].x0[2] = 0.0;
  chn[0].xbnd[4] = -5.;
  chn[0].xbnd[5] = 5.;

  chn[0].x0[3] = 0.0;
  chn[0].xbnd[6] = -5.;
  chn[0].xbnd[7] = 5.;

  chn[0].x0[4] = 1.5;
  chn[0].xbnd[8] = -25.;
  chn[0].xbnd[9] = 25.;

  chn[0].x0[5] = -5.;
  chn[0].xbnd[10] = -25.;
  chn[0].xbnd[11] = 25.;

  chn[0].x0[6] = 1.5;
  chn[0].xbnd[12] = -25.;
  chn[0].xbnd[13] = 25.;

  chn[0].x0[7] = -5.;
  chn[0].xbnd[14] = -25.;
  chn[0].xbnd[15] = 25.;

  chn[0].x0[8] = 1.5;
  chn[0].xbnd[16] = -25.;
  chn[0].xbnd[17] = 25.;

  chn[0].x0[9] = -5.;
  chn[0].xbnd[18] = -25.;
  chn[0].xbnd[19] = 25.;

  chn[0].x0[10] = 1.5;
  chn[0].xbnd[20] = -25.;
  chn[0].xbnd[21] = 25.;

  chn[0].x0[11] = -5.;
  chn[0].xbnd[22] = -25.;
  chn[0].xbnd[23] = 25.;

  chn[0].x0[12] = 1.5;
  chn[0].xbnd[24] = -25.;
  chn[0].xbnd[25] = 25.;

  chn[0].x0[13] = -5.;
  chn[0].xbnd[26] = -25.;
  chn[0].xbnd[27] = 25.;

  chn[0].x0[14] = 1.5;
  chn[0].xbnd[28] = -25.;
  chn[0].xbnd[29] = 25.;

  chn[0].x0[15] = -5.;
  chn[0].xbnd[30] = -25.;
  chn[0].xbnd[31] = 25.;

  chn[0].x0[16] = 1.5;
  chn[0].xbnd[32] = -25.;
  chn[0].xbnd[33] = 25.;

  chn[0].x0[17] = -5.;
  chn[0].xbnd[34] = -25.;
  chn[0].xbnd[35] = 25.;

  chn[0].x0[18] = 1.5;
  chn[0].xbnd[36] = -25.;
  chn[0].xbnd[37] = 25.;

  chn[0].x0[19] = -5.;
  chn[0].xbnd[38] = -25.;
  chn[0].xbnd[39] = 25.;

  chn[0].x0[20] = 0.2;
  chn[0].xbnd[40] = 0.;
  chn[0].xbnd[41] = 25.;

  initializeChain ( cdp, chn, mdl, spc );

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

  if ( vrb ) {
    printf ( "      ... >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done!\n" );
  }

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Time to generate: %3.1f ms\n", chn[0].time );
    printf ( "\n" );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  averagedAutocorrelationFunction ( cdp, chn );

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Autocorrelation time window -- %i\n", chn[0].mmm );
    printf ( " Autocorrelation time -- %.8E\n", chn[0].atcTime );
    printf ( " Autocorrelation time threshold -- %.8E\n", chn[0].nst / 5e1f );
    printf ( " Effective number of independent samples -- %.8E\n", chn[0].nwl * chn[0].nst / chn[0].atcTime );
    printf ( " Time to compute acor time: %3.1f ms\n", chn[0].time );
    printf ( "\n" );
  }

  /* Write results to a file */
  simpleWriteDataFloat ( "Autocor.out", chn[0].nst, chn[0].atcrrFnctn );
  simpleWriteDataFloat ( "AutocorCM.out", chn[0].nst, chn[0].cmSmAtCrrFnctn );
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
