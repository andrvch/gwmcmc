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
    printf ( " Device ID: %d\n", cdp[0].dev );
    printf ( " Device name: %s\n", cdp[0].prop.name );
    printf ( " Driver API: v%d \n", cdp[0].driverVersion[0] );
    printf ( " Runtime API: v%d \n", cdp[0].runtimeVersion[0] );
    printf ( "\n" );
  }

  Chain chn[1];
  chn[0].name = argv[2];
  chn[0].nwl = atoi ( argv[3] );
  chn[0].nst = atoi ( argv[4] );

  allocateChainForAutoCorr ( chn );

  simpleReadDataFloat ( chn[0].name, chn[0].chnFnctn );

  /*
  for ( int i = 0; i < chn[0].nwl*chn[0].nst; i++ ) {
    printf ( " %i ", i );
    printf ( " %2.4f ", chn[0].chnFnctn[i] );
    printf ( "\n" );
  }*/

  cudaEventRecord ( cdp[0].start, 0 );

  atcrrltnfnctn ( cdp, chn );

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( " Autocorrelation time window: %i\n", chn[0].mmm );
    printf ( " Autocorrelation time: %.8E\n", chn[0].atcTime );
    printf ( " Autocorrelation time threshold: %.8E\n", chn[0].nst / 5e1f );
    printf ( " Effective number of independent samples: %.8E\n", chn[0].nwl * chn[0].nst / chn[0].atcTime );
    printf ( " Time to compute acor time: %3.1f ms\n", chn[0].time );
    printf ( "\n" );
  }

  /* Write results to a file */
  printf ( " Write results to the host memory and clean up ... \n" );
  simpleWriteDataFloat ( "acrr.out", chn[0].nst, chn[0].atcrrFnctn );
  simpleWriteDataFloat ( "acrrcm.out", chn[0].nst, chn[0].cmSmAtCrrFnctn );

  destroyCuda ( cdp );
  freeChainForAutoCorr ( chn );

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

  printf ( " Done!\n" );

  return 0;
}

#endif // _GWMCMCCUDA_CU_
