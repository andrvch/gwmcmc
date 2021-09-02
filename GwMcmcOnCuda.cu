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
  chn[0].indx = atoi ( argv[5] );
  chn[0].dlt = 0.2E-5;

  Image img[NIMG];
  float pixdim = atoi ( argv[6] );
  for ( int i = 0; i < NIMG; i++ ) {
    img[i].imdim = pixdim;
    img[i].nx = pixdim;
    img[i].ny = pixdim;
    img[i].pix = 1.;
  }

  for ( int i = 0; i < NIMG; i++ ) {
    img[i].psffl = argv[7+2*i];
    img[i].datafl = argv[7+2*i+1];
  }

  img[0].xref = 4219.1376;
  img[0].yref = 4113.3950;
  img[1].xref = 3869.3410;
  img[1].yref = 3946.4976;
  img[2].xref = 4213.6589;
  img[2].yref = 4286.8925;
  img[3].xref = 4118.4548;
  img[3].yref = 4133.1050;

  img[4].xref = 4211.8631;
  img[4].yref = 4091.1163;
  img[5].xref = 3861.809;
  img[5].yref = 3924.7092;
  img[6].xref = 4205.8589;
  img[6].yref = 4264.7592;
  img[7].xref = 4110.6743;
  img[7].yref = 4110.2977;


  chn[0].dim = 3 * ( NIMG / 2 ) + ( NIMG / 2 ) + 3 + 2;
  //printf ( " chain dimension = %i \n ", chn[0].dim );

  allocateChain ( chn );
  allocateImage ( chn, img );

  for ( int i = 0; i < NIMG; i++ ) {
    chn[0].phr[2*i] = img[i].xref;
    chn[0].phr[2*i+1] = img[i].yref;
  }

  for ( int i = 0; i < NIMG; i++ ) {
    img[i].idx = i;
  }

  // set up starting values:
  for ( int i = 0; i < NIMG/2; i++ ) {
    chn[0].x0[3*i] = 0.;
    chn[0].x0[3*i+1] = 0.;
    chn[0].x0[3*i+2] = 0.04;
  }

  for ( int i = 0; i < NIMG/2; i++ ) {
    chn[0].x0[3*NIMG/2+i] = 0.04;
  }

  chn[0].x0[3*NIMG/2+NIMG/2] = 7.3;
  chn[0].x0[3*NIMG/2+NIMG/2+1] = 22.3;
  chn[0].x0[3*NIMG/2+NIMG/2+2] = 0.0;

  chn[0].x0[3*NIMG/2+NIMG/2+3] = 0.0;
  chn[0].x0[3*NIMG/2+NIMG/2+4] = 0.0;

  // set up boundaries:
  for ( int i = 0; i < NIMG/2; i++ ) {
    chn[0].x0bn[6*i] = -5.;
    chn[0].x0bn[6*i+1] = 5.;
    chn[0].x0bn[6*i+2] = -5.;
    chn[0].x0bn[6*i+3] = 5.;
    chn[0].x0bn[6*i+4] = 0.0;
    chn[0].x0bn[6*i+5] = 10000.;
  }

  for ( int i = 0; i < NIMG/2; i++ ) {
    chn[0].x0bn[2*(3*NIMG/2)+2*i] = 0.0;
    chn[0].x0bn[2*(3*NIMG/2)+2*i+1] = 10000.;
  }

  chn[0].x0bn[2*(3*NIMG/2)+NIMG] = -100.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+1] = 100.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+2] = -100.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+3] = 100.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+4] = -PI;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+5] = PI;

  chn[0].x0bn[2*(3*NIMG/2)+NIMG+6] = -5.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+7] = 5.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+8] = -5.;
  chn[0].x0bn[2*(3*NIMG/2)+NIMG+9] = 5.;

  for ( int i = 0; i < NIMG; i++ ) {
    simpleReadDataFloat ( img[i].psffl, img[i].psf );
    simpleReadDataFloat ( img[i].datafl, img[i].img );
  }

  /*
  printf ( "Input psf file:" );
  printf ( "\n" );
  printf ( "\n" );
  for ( int i = 0; i < img[0].nx*img[0].ny; i ++ ) {
    printf ( " %4.4f ", img[0].psf[i] );
  }
  printf ( "\n" );
  printf ( "\n" );
  */

  initializeChain ( cdp, chn, img );

  /*
  cudaDeviceSynchronize ();

  printf ( "Initial walkers:" );
  printf ( "\n" );
  printf ( "\n" );
  for ( int j = 0; j < chn[0].nwl; j ++ ) {
    for ( int i = 0; i < chn[0].dim; i ++ ) {
      printf ( " %4.4f " , chn[0].xx[i+j*chn[0].dim] );
    }
    printf ( "\n" );
    printf ( "\n" );
  }
  printf ( "Initial shifted psf's:" );
  printf ( "\n" );
  printf ( "\n" );
  for ( int j = 0; j < chn[0].nwl; j ++ ) {
    printf ( " walk num %i :\n", j );
    for ( int i = 0; i < chn[0].nx*chn[0].ny; i ++ ) {
      printf ( " %4.4f " , chn[0].pp[i+j*chn[0].nx*chn[0].ny] );
      //printf ( " %i " , chn[0].ww[i+j*chn[0].nx*chn[0].ny] );
    }
    printf ( "\n" );
    printf ( "\n" );
  }
  printf ( "Initial stat:" );
  printf ( "\n" );
  printf ( "\n" );
  for ( int j = 0; j < chn[0].nwl; j ++ ) {
    printf ( " walk num %i :\n", j );
    for ( int i = 0; i < chn[0].nx*chn[0].ny; i ++ ) {
      printf ( " %4.4f " , chn[0].sstt[i+j*chn[0].nx*chn[0].ny] );
      //printf ( " %i " , chn[0].ww[i+j*chn[0].nx*chn[0].ny] );
    }
    printf ( "\n" );
    printf ( "\n" );
  }
  printf ( "Initial total stat:" );
  printf ( "\n" );
  printf ( "\n" );
  for ( int j = 0; j < chn[0].nwl; j ++ ) {
    printf ( " walk num %i :\n", j );
    //for ( int i = 0; i < chn[0].nx*chn[0].ny; i ++ ) {
    printf ( " %4.4f " , chn[0].stt[j] );
      //printf ( " %i " , chn[0].ww[i+j*chn[0].nx*chn[0].ny] );
    //}
    printf ( "\n" );
    printf ( "\n" );
  }*/

  if ( vrb ) {
    printf ( " Start ... \n" );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  initializeRandomForStreach ( cdp, chn );
  //initializeRandomForWalk ( cdp, chn );
  //initializeRandomForMetropolis ( cdp, chn );

  chn[0].ist = 0;
  while ( chn[0].ist < chn[0].nst ) {
    /*metropolisMove ( cdp, chn );
    statisticMetropolis ( cdp, chn );
    metropolisUpdate ( cdp, chn );*/
    chn[0].isb = 0;
    while ( chn[0].isb < 2 ) {
      //walkMove ( cdp, chn );
      streachMove ( cdp, chn );
      //metropolisMove ( cdp, chn );
      /*
      cudaDeviceSynchronize ();
      printf ( "Proposed walkers:" );
      printf ( "\n" );
      printf ( "\n" );
      for ( int j = 0; j < chn[0].nwl/2; j ++ ) {
        for ( int i = 0; i < chn[0].dim; i ++ ) {
          printf ( " %4.4f " , chn[0].xx1[i+j*chn[0].dim] );
        }
        printf ( "\n" );
        printf ( "\n" );
      }
      printf ( "Proposed shifted psf's:" );
      printf ( "\n" );
      printf ( "\n" );
      for ( int j = 0; j < chn[0].nwl/2; j ++ ) {
        printf ( " walk num %i :\n", j );
        for ( int i = 0; i < chn[0].nx*chn[0].ny; i ++ ) {
          printf ( " %4.4f " , chn[0].pp[i+j*chn[0].nx*chn[0].ny] );
          //printf ( " %i " , chn[0].ww[i+j*chn[0].nx*chn[0].ny] );
        }
        printf ( "\n" );
        printf ( "\n" );
      }*/
      //printMetropolisMove ( chn );
      statistic ( cdp, chn, img );
      //statisticMetropolis ( cdp, chn );
      /*
      cudaDeviceSynchronize ();
      printf ( "Proposed stat:" );
      printf ( "\n" );
      printf ( "\n" );
      for ( int j = 0; j < chn[0].nwl/2; j ++ ) {
        printf ( " walk num %i :\n", j );
        for ( int i = 0; i < chn[0].nx*chn[0].ny; i ++ ) {
          printf ( " %4.4f " , chn[0].sstt1[i+j*chn[0].nx*chn[0].ny] );
          //printf ( " %i " , chn[0].ww[i+j*chn[0].nx*chn[0].ny] );
        }
        printf ( "\n" );
        printf ( "\n" );
      }
      //printMetropolisMove ( chn );
      //printMove ( chn );
      //walkUpdate ( cdp, chn );
      */
      streachUpdate ( cdp, chn );
      //metropolisUpdate ( cdp, chn );
      /*
      cudaDeviceSynchronize ();
      printf ( "Updated walkers:" );
      printf ( "\n" );
      printf ( "\n" );
      for ( int j = 0; j < chn[0].nwl; j ++ ) {
        for ( int i = 0; i < chn[0].dim; i ++ ) {
          printf ( " %4.4f " , chn[0].xx[i+j*chn[0].dim] );
        }
        printf ( "\n" );
        printf ( "\n" );
      }
      printf ( "Updated stat:" );
      printf ( "\n" );
      printf ( "\n" );
      for ( int j = 0; j < chn[0].nwl; j ++ ) {
        printf ( " walk num %i :\n", j );
        //for ( int i = 0; i < chn[0].nx*chn[0].ny; i ++ ) {
        printf ( " %4.4f " , chn[0].stt[j] );
          //printf ( " %i " , chn[0].ww[i+j*chn[0].nx*chn[0].ny] );
        //}
        printf ( "\n" );
        printf ( "\n" );
      } */
      //printMetropolisUpdate ( chn );
      //printUpdate ( chn );
      chn[0].isb += 1;
    }
    saveCurrent ( chn );
    chn[0].ist += 1;
  }

  if ( vrb ) {
    printf ( " Done!\n" );
    printf ( "\n" );
  }

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( " Time to generate: %3.1f ms\n", chn[0].time );
  }

  cudaEventRecord ( cdp[0].start, 0 );

  averagedAutocorrelationFunction ( cdp, chn );

  cudaEventRecord ( cdp[0].stop, 0 );
  cudaEventSynchronize ( cdp[0].stop );
  cudaEventElapsedTime ( &chn[0].time, cdp[0].start, cdp[0].stop );

  if ( vrb ) {
    printf ( " Autocorrelation time window: %i\n", chn[0].mmm );
    printf ( " Autocorrelation time: %.8E\n", chn[0].atcTime );
    printf ( " Autocorrelation time threshold: %.8E\n", chn[0].nwl * chn[0].nst / 5e1f );
    printf ( " Effective number of independent samples: %.8E\n", chn[0].nwl * chn[0].nst / chn[0].atcTime );
    printf ( " Time to compute acor time: %3.1f ms\n", chn[0].time );
    printf ( "\n" );
  }

  /* Write results to a file */
  printf ( " Write results to the host memory and clean up ... \n" );
  simpleWriteDataFloat ( "Autocor.out", chn[0].nst, chn[0].atcrrFnctn );
  simpleWriteDataFloat ( "AutocorCM.out", chn[0].nst, chn[0].cmSmAtCrrFnctn );
  writeChainToFile ( chn[0].name, chn[0].indx, chn[0].dim, chn[0].nwl, chn[0].nst, chn[0].smpls, chn[0].stat );

  destroyCuda ( cdp );
  freeChain ( chn );
  freeImage ( img );

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
