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

  // cuda context:
  Cupar cdp[1];
  cdp[0].dev = atoi ( argv[1] );
  initializeCuda ( cdp );

  // print cuda info:
  if ( vrb ) {
    printf ( "\n" );
    printf ( ".................................................................\n" );
    printf ( " CUDA device ID: %d\n", cdp[0].dev );
    printf ( " CUDA device Name: %s\n", cdp[0].prop.name );
    printf ( " Driver API: v%d \n", cdp[0].driverVersion[0] );
    printf ( " Runtime API: v%d \n", cdp[0].runtimeVersion[0] );
  }

  // model:
  Model mdl[1];
  InitializeModel ( mdl );

  // chain:
  Chain chn[1];
  chn[0].name = argv[2];
  chn[0].nwl = atoi ( argv[3] );
  chn[0].nst = atoi ( argv[4] );
  chn[0].indx = atoi ( argv[5] );
  chn[0].dim = NPRS;
  chn[0].dim1 = chn[0].dim + 3;
  chn[0].dlt = 1.E-3;
  chn[0].nkb = 100;

  // spectrum:
  Spectrum spc[NSPCTR];
  for ( int i = 0; i < NSPCTR; i++ ) {
    spc[i].name = argv[6+i];
  }

  for ( int i = 0; i < NSPCTR; i++ ) {
    spc[i].lwrNtcdEnrg = ( float ) atof ( argv[6+NSPCTR] );
    spc[i].hghrNtcdEnrg = ( float ) atof ( argv[6+NSPCTR+1] );
  }

  SpecInfo ( vrb, spc );
  SpecAlloc ( chn, spc );
  SpecData ( cdp, vrb, mdl, spc );

  allocateChain ( chn );

  // set starting parameters and boundaries:
  chn[0].x0[0] = 5.7;
  chn[0].xbnd[0] = 5.5;
  chn[0].xbnd[1] = 6.5;

  chn[0].x0[1] = -6.;
  chn[0].xbnd[2] = -25.;
  chn[0].xbnd[3] = 25.;

  chn[0].x0[2] = 0.1;
  chn[0].xbnd[4] = 0.0;
  chn[0].xbnd[5] = 25.;

  // initialize chain:
  initializeChain ( cdp, chn );
  if ( chn[0].indx == 0 ) {
    modelStatistic0 ( cdp, mdl, chn, spc );
  }

  /*
  cudaDeviceSynchronize ();

  for ( int j = 0; j < spc[0].nmbrOfNtcdBns; j++ ) {
    printf ( " %.8E ", spc[0].srcGrp[j] );
  }
  printf ( "\n" );
  printf ( "\n" );
  printf ( "\n" );

  for ( int i = 0; i < chn[0].nwl; i++ ) {
    for ( int j = 0; j < spc[0].nmbrOfNtcdBns; j++ ) {
      printf ( " %.8E ", spc[0].chnnlSttstcs[j+i*spc[0].nmbrOfNtcdBns] );
    }
    printf ( "\n" );
    printf ( "\n" );
    printf ( "\n" );

    printf ( " %.8E ", chn[0].stt[i] );

    printf ( "\n" );
    printf ( "\n" );
    printf ( "\n" );
  }

  printf ( "\n" );
  printf ( "\n" );
  printf ( "\n" );

  for ( int i = 0; i < chn[0].nwl; i++ ) {
    for ( int j = 0; j < spc[0].nmbrOfNtcdBns; j++ ) {
      printf ( " %.8E ", spc[0].flddMdlFlxs[j+i*spc[0].nmbrOfNtcdBns] );
    }
    printf ( "\n" );
    printf ( "\n" );
    printf ( "\n" );
    printf ( "\n" );
  }
  */

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Start ...                                                  \n" );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  // initialize random numbers:
  initializeRandomForStreach ( cdp, chn );

  // start chain:
  chn[0].ist = 0;
  while ( chn[0].ist < chn[0].nst ) {
    chn[0].isb = 0;
    while ( chn[0].isb < 2 ) {
      streachMove ( cdp, chn );

      modelStatistic1 ( cdp, mdl, chn, spc );

      cudaDeviceSynchronize ();

      for ( int i = 0; i < chn[0].nwl/2; i++ ) {
        for ( int j = 0; j < spc[0].nmbrOfNtcdBns; j++ ) {
          printf ( " %.8E ", spc[0].chnnlSttstcs[j+i*spc[0].nmbrOfNtcdBns] );
        }
        printf ( "\n" );
        printf ( "\n" );
        printf ( "\n" );

        printf ( " %.8E ", chn[0].stt1[i] );

        printf ( "\n" );
        printf ( "\n" );
        printf ( "\n" );
      }

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
    printf ( " Autocorrelation time threshold          -- %.8E\n", chn[0].nwl * chn[0].nst / 5e1f );
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
  //chn[0].xx[10] = 0.05;
  //chn[0].xx[11] = 100.;
  chn[0].didi[0] = chn[0].skbin[chn[0].dim];
  printf ( " %2.2f ", chn[0].didi[0] );
  chn[0].stt[0] = chn[0].skbin[chn[0].dim+1];
  printf ( " %2.2f ", chn[0].stt[0] );
  chn[0].chi[0] = chn[0].skbin[chn[0].dim+2];
  printf ( " %2.2f ", chn[0].chi[0] );
  printf ( "\n" );

  modelStatistic00 ( cdp, mdl, chn, spc );

  cudaDeviceSynchronize ();

  /* Write results to a file */
  simpleWriteDataFloat ( "Autocor.out", chn[0].nst, chn[0].atcrrFnctn );
  simpleWriteDataFloat ( "AutocorCM.out", chn[0].nst, chn[0].cmSmAtCrrFnctn );
  writeSpectraToFile ( chn[0].name, spc );
  writeKdeToFile ( chn[0].name, chn[0].dim1, chn[0].nkb, chn[0].kbin, chn[0].kdePdf );
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
