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

  Spectrum spc[NSPCTR];
  Spectrum bkg[NSPCTR];
  for ( int i = 0; i < NSPCTR; i++ ) {
    spc[i].name = argv[2+2*i];
    bkg[i].name = argv[2+2*i+1];
  }

  chn[0].name = argv[NSPCTR11+2];
  chn[0].nwl = atoi ( argv[NSPCTR11+3] );
  chn[0].nst = atoi ( argv[NSPCTR11+4] );
  chn[0].indx = atoi ( argv[NSPCTR11+5] );
  chn[0].dim = NPRS;
  chn[0].dim1 = chn[0].dim + 3;
  chn[0].dlt = 1.E-4;
  chn[0].nkb = 100;

  for ( int i = 0; i < NSPCTR; i++ ) {
    spc[i].lwrNtcdEnrg = ( float ) atof ( argv[NSPCTR11+6] );
    spc[i].hghrNtcdEnrg = ( float ) atof ( argv[NSPCTR11+7] );
  }

  Model mdl[1];
  InitializeModel ( mdl );

  SpecInfo ( vrb, spc );
  SpecInfo ( vrb, bkg );

  for ( int i = 0; i < NSPCTR; i++ ) {
    bkg[i].nmbrOfBns = spc[i].nmbrOfBns;
  }

  SpecAlloc ( chn, spc );
  SpecAlloc ( chn, bkg );

  SpecData ( cdp, vrb, mdl, spc, bkg );


  destroyCuda ( cdp );
  //freeChain ( chn );
  FreeModel ( mdl );

  FreeSpec ( spc );
  FreeSpec ( bkg );

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
