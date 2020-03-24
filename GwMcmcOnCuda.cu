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
  chn[0].dfl = argv[2];
  chn[0].name = argv[3];
  chn[0].nwl = atoi ( argv[4] );
  chn[0].nst = atoi ( argv[5] );
  chn[0].indx = atoi ( argv[6] );
  chn[0].dim = 2;
  chn[0].dlt = 1.E-6;
  chn[0].nbm = atoi ( argv[7] );

  readTimesInfo ( chn[0].dfl, &chn[0].nph, &chn[0].exptm );
  //chn[0].nph = 372383;
  //chn[0].nph = 272217;

  allocateChain ( chn );
  allocateTimes ( chn );

  readTimesData ( chn[0].dfl, chn[0].nph, chn[0].atms );

  simpleReadDataFloat ( chn[0].dfl, chn[0].atms );

  chn[0].scale = chn[0].nph * logf ( chn[0].nbm ) + chn[0].nph * logf ( chn[0].nph * 1. ) - logf ( chn[0].nph * 1. ) - ( chn[0].nph + chn[0].nbm - 1. ) * logf ( ( chn[0].nph + chn[0].nbm - 1. ) * 1. ) + logf ( ( chn[0].nph + chn[0].nbm - 1. ) * 1. );

  int sumsum = 0;
  for ( int i = 0; i < chn[0].nbm-2; i++ ) {
    sumsum += logf ( i + 2 );
  }
  chn[0].scale = chn[0].scale + sumsum;

  //for ( int i = 0; i < chn[0].dim; i++ ) {
  chn[0].x0[0] = ( float ) atof ( argv[8] ); //592.4215; //1.68799e-3; //3.362332;
  chn[0].x0[1] = 1./chn[0].nbm/2.;
  //}

  //for ( int i = 0; i < chn[0].dim; i++ ) {
  chn[0].xbnd[0] = 2.62;
  chn[0].xbnd[1] = 2.70;
  chn[0].xbnd[2] = 0.0;
  chn[0].xbnd[3] = 1./chn[0].nbm;
  //}

  initializeChain ( cdp, chn );

  if ( vrb ) {
    printf ( ".................................................................\n" );
    printf ( " Start ...                                                  \n" );
  }

  chn[0].sigma[0] = ( float ) atof ( argv[9] );
  chn[0].sigma[1] = 1. / chn[0].nbm / 10.;

  //cudaDeviceSynchronize ();
  //printMetropolisMove ( chn );

  cudaEventRecord ( cdp[0].start, 0 );

  //initializeRandomForStreach ( cdp, chn );
  //initializeRandomForWalk ( cdp, chn );
  initializeRandomForMetropolis ( cdp, chn );

  chn[0].ist = 0;
  while ( chn[0].ist < chn[0].nst ) {
    /*metropolisMove ( cdp, chn );
    statisticMetropolis ( cdp, chn );
    metropolisUpdate ( cdp, chn );*/
    chn[0].isb = 0;
    while ( chn[0].isb < chn[0].dim ) {
      //walkMove ( cdp, chn );
      //streachMove ( cdp, chn );
      metropolisMove ( cdp, chn );
      //cudaDeviceSynchronize ();
      //printMetropolisMove ( chn );
      //statistic ( cdp, chn );
      //statisticMetropolis ( cdp, chn );
      modelStatistic1 ( cdp, chn );
      //cudaDeviceSynchronize ();
      //printMetropolisMove ( chn );
      //printMove ( chn );
      //walkUpdate ( cdp, chn );
      //streachUpdate ( cdp, chn );
      metropolisUpdate ( cdp, chn );
      //cudaDeviceSynchronize ();
      //printMetropolisUpdate ( chn );
      //printUpdate ( chn );
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
  writeChainToFile ( chn[0].name, chn[0].indx, chn[0].dim, chn[0].nwl, chn[0].nst, chn[0].nbm, chn[0].smpls, chn[0].stat, chn[0].numbers, chn[0].prior );

  destroyCuda ( cdp );
  freeChain ( chn );
  freeTimes ( chn );

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
