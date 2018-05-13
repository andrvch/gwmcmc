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
#include "GwMcmcStructuresFunctionsAndKernels.cuh"

/* Functions and Kernels: */
__host__ __device__ int PriorCondition ( const Walker wlkr )
{
    int indx, cndtn = 1;
    indx = 0;
    cndtn = cndtn * ( wlkr.par[indx] > 0 ) * ( wlkr.par[indx] < 5.5 );
    indx = 1; // pl normalization
    cndtn = cndtn * ( wlkr.par[indx] > -15. );
    indx = 2; // Temperature
    cndtn = cndtn * ( wlkr.par[indx] > 0.03 ) * ( wlkr.par[indx] < 1. );
    indx = 3; // Radi
    //cndtn = cndtn * ( wlkr.par[indx] > 0.0 );
    indx = 4; // Distance
    cndtn = cndtn * ( wlkr.par[indx] < 3.3 ) * ( wlkr.par[indx] > 1. );
    indx = NHINDX; // Hydrogen column density
    cndtn = cndtn * ( wlkr.par[indx] > 0 );
    return cndtn;
}

__host__ __device__ float PriorStatistic ( const Walker wlkr, const int cndtn, const float mNh, const float sNh )
{
    int indx = NHINDX; // Hydrogen column density
    float prr = 0, sum = 0, mean = 0, sigma = 0.06;
    float theta = powf ( sNh, 2 ) / mNh;
    float kk = mNh / theta;
    sum = sum + ( kk - 1 ) * logf ( wlkr.par[indx] ) - wlkr.par[indx] / theta;
    //sum = sum + powf ( ( wlkr.par[indx] - mNh ) / sNh, 2 );
    indx = NHINDX + 1;
    while ( indx < NPRS )
    {
        sum = sum + powf ( ( wlkr.par[indx] - mean ) / sigma, 2 );
        indx += 1;
    }
    if ( cndtn ) { prr = sum; } else { prr = INF; }
    return prr;
}

__global__ void AssembleArrayOfAbsorptionFactors ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int nmbrOfElmnts,
                                                   const float *crssctns, const float *abndncs, const int *atmcNmbrs, const Walker *wlkrs,
                                                   float *absrptnFctrs )
{
    int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;

    int ttIndx = enIndx + wlIndx * nmbrOfEnrgChnnls;
    int elIndx, effElIndx, crIndx, prIndx;
    float xsctn, clmn, nh;
    if ( ( enIndx < nmbrOfEnrgChnnls ) && ( wlIndx < nmbrOfWlkrs ) )
    {
        elIndx = 0;
        prIndx = elIndx + NHINDX;
        crIndx = elIndx + enIndx * nmbrOfElmnts;
        effElIndx = atmcNmbrs[elIndx] - 1;
        nh = wlkrs[wlIndx].par[prIndx] * 1.E22;
        clmn = abndncs[effElIndx];
        xsctn = clmn * crssctns[crIndx];
        elIndx = 1;
        while ( elIndx < nmbrOfElmnts )
        {
            prIndx = elIndx + NHINDX;
            crIndx = elIndx + enIndx * nmbrOfElmnts;
            effElIndx = atmcNmbrs[elIndx] - 1;
            clmn = abndncs[effElIndx]; // * powf ( 10, wlkrs[wlIndx].par[prIndx] );
            xsctn = xsctn + clmn * crssctns[crIndx];
            elIndx += 1;
        }
        absrptnFctrs[ttIndx] = expf ( - nh * xsctn );
    }
}

__global__ void AssembleArrayOfModelFluxes ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls,
                                             const float *enrgChnnls, const float *arfFctrs, const float *absrptnFctrs, const Walker *wlkrs,
                                             float *mdlFlxs )
{
    int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
    int ttIndx = enIndx + wlIndx * nmbrOfEnrgChnnls;
    if ( ( enIndx < nmbrOfEnrgChnnls ) && ( wlIndx < nmbrOfWlkrs ) )
    {
//        mdlFlxs[ttIndx] =  arfFctrs[enIndx] * absrptnFctrs[ttIndx] * BlackBody ( wlkrs[wlIndx].par[0], wlkrs[wlIndx].par[1], enrgChnnls[enIndx], enrgChnnls[enIndx+1] );
//        mdlFlxs[ttIndx] =  arfFctrs[enIndx] * absrptnFctrs[ttIndx] * PowerLaw ( wlkrs[wlIndx].par[0], wlkrs[wlIndx].par[1], enrgChnnls[enIndx], enrgChnnls[enIndx+1] );
        mdlFlxs[ttIndx] =  arfFctrs[enIndx] * absrptnFctrs[ttIndx] * ( PowerLaw ( wlkrs[wlIndx].par[0], wlkrs[wlIndx].par[1], enrgChnnls[enIndx], enrgChnnls[enIndx+1] )
        + BlackBody ( wlkrs[wlIndx].par[2], wlkrs[wlIndx].par[3], enrgChnnls[enIndx], enrgChnnls[enIndx+1] ) );

    }
}

/**
 * Host main routine
 */
int main ( int argc, char *argv[] )
{
    /* Set up initial parameters: */
    const int devId = atoi( argv[1] );
    const char *spcFl = argv[2];
    const char *spcLst[NSPCTR] = { spcFl, spcFl };
    const char *thrdNm = argv[3];
    const int nmbrOfWlkrs = atoi ( argv[4] );
    const int nmbrOfStps = atoi ( argv[5] );
    const int thrdIndx = atoi ( argv[6] );
    const float lwrNtcdEnrg = 0.3;
    const float hghrNtcdEnrg = 8.0;
    const float dlt = 1.E-4;
    const float phbsPwrlwInt[NPRS] = { 1.1, log10f ( 9.E-6 ), 0.1, -3., log10f ( 8E2 ), 0.15 };

    int incxx = INCXX, incyy = INCYY;
    float alpha = ALPHA, beta = BETA;
    dim3 dimBlock ( THRDSPERBLCK, THRDSPERBLCK );
    const int verbose = 1;

    /* 1 ) Initialize */
    Cuparam cdp[NSPCTR];
    Model mdl[NSPCTR];
    Chain chn[NSPCTR];
    Spectrum spc[NSPCTR];

    InitializeCuda ( devId, cdp );
    InitializeModel ( mdl );
    InitializeChain ( thrdNm, thrdIndx, nmbrOfWlkrs, nmbrOfStps, phbsPwrlwInt, cdp[0].curandGnrtrHst, dlt, chn );
    InitializeSpectra ( spcLst, cdp[0].cusparseHandle, cdp[0].cusparseStat, cdp[0].cublasHandle, cdp[0].cublasStat, verbose, nmbrOfWlkrs, lwrNtcdEnrg, hghrNtcdEnrg, mdl[0].sgFlg, mdl[0].atmcNmbrs, spc );

    if ( thrdIndx > 0 )
    {
        /* Initialize walkers and statistics from last chain */
        InitializeWalkersAndStatisticsFromLastChain <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, chn[0].lstWlkrsAndSttstcs, chn[0].wlkrs, chn[0].sttstcs );
    }
    else if ( thrdIndx == 0 )
    {
        /* 1 ) Generate uniformly distributed floating point values between 0.0 and 1.0, chn[0].rndmVls[nmbrOfWlkrs] (cuRand) */
        curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].rndmVls, nmbrOfWlkrs );
        /* 2 ) Initialize walkers, actlWlkrs[nmbrOfWlkrs] */
        InitializeWalkersAtRandom <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, dlt, chn[0].strtngWlkr, chn[0].rndmVls, chn[0].wlkrs );
        /* 3 ) Assemble array of absorption factors, spc[0].absrptnFctrs[nmbrOfWlkrs*spc[0].nmbrOfEnrgChnnls] */
        AssembleArrayOfAbsorptionFactors <<< Grid ( spc[0].nmbrOfEnrgChnnls, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spc[0].nmbrOfEnrgChnnls, ATNMR, spc[0].crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].wlkrs, spc[0].absrptnFctrs );
        /* 4 a ) Assemble array of nsa fluxes */
        //BilinearInterpolation <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spc[0].nmbrOfEnrgChnnls, 2, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spc[0].enrgChnnls, chn[0].wlkrs, spc[0].mdlFlxs );
        /* 4 ) Assemble array of model fluxes, spc[0].mdlFlxs[nmbrOfWlkrs*spc[0].nmbrOfEnrgChnnls] */
        AssembleArrayOfModelFluxes <<< Grid ( spc[0].nmbrOfEnrgChnnls, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spc[0].nmbrOfEnrgChnnls, spc[0].enrgChnnls, spc[0].arfFctrs, spc[0].absrptnFctrs, chn[0].wlkrs, spc[0].mdlFlxs );
        /* 5 ) Fold model fluxes with RMF, spc[0].flddMdlFlxs[nmbrOfWlkrs*spc[0].nmbrOfChnnls] (cuSparse) */
        cdp[0].cusparseStat = cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[0].nmbrOfChnnls, nmbrOfWlkrs, spc[0].nmbrOfEnrgChnnls, spc[0].nmbrOfRmfVls, &alpha, cdp[0].MatDescr, spc[0].rmfVls, spc[0].rmfPntr, spc[0].rmfIndx, spc[0].mdlFlxs, spc[0].nmbrOfEnrgChnnls, &beta, spc[0].flddMdlFlxs, spc[0].nmbrOfChnnls );
        if ( cdp[0].cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Matrix-matrix multiplication failed yes " ); return 1; }
        /* 6 ) Assemble array of channel statistics, spc[0].chnnlSttstcs[nmbrOfWlkrs*spc[0].nmbrOfChnnls] */
        AssembleArrayOfChannelStatistics <<< Grid ( spc[0].nmbrOfChnnls, nmbrOfWlkrs ), dimBlock >>> ( nmbrOfWlkrs, spc[0].nmbrOfChnnls, spc[0].srcExptm, spc[0].bckgrndExptm, spc[0].srcCnts, spc[0].bckgrndCnts, spc[0].flddMdlFlxs, spc[0].chnnlSttstcs );
        /* 7 ) Sum up channel statistics, actlSttstcs[nmbrOfWlkrs] (cuBlas) */
        cdp[0].cublasStat = cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[0].nmbrOfChnnls, nmbrOfWlkrs, &alpha, spc[0].chnnlSttstcs, spc[0].nmbrOfChnnls, spc[0].ntcdChnnls, incxx, &beta, chn[0].sttstcs, incyy );
        if ( cdp[0].cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed 0 " ); return 1; }
    }

    printf ( ".................................................................\n" );
    printf ( " Start ...                                                  \n" );

    cudaEventRecord ( cdp[0].start, 0 );

    /* Generate uniformly distributed floating point values between 0.0 and 1.0, chn[0].rndmVls[3*nmbrOfWlkrs / 2*nmbrOfStps] (cuRand) */
    curandGenerateUniform ( cdp[0].curandGnrtr, chn[0].rndmVls, 3 * nmbrOfWlkrs / 2 * nmbrOfStps );

    /* Start MCMC !!!!! */
    int stpIndx = 0, sbstIndx;
    while ( stpIndx < nmbrOfStps )
    {
        /* Initialize subset index */
        sbstIndx = 0;
        /* Iterate over two subsets */
        while ( sbstIndx < 2 )
        {
            /* 1 ) Generate Z values and proposed walkers, chn[0].zRndmVls[nmbrOfWlkrs / 2], chn[0].prpsdWlkrs[nmbrOfWlkrs / 2] */
            GenerateProposal <<< Blocks ( nmbrOfWlkrs / 2 ), THRDSPERBLCK >>> ( nmbrOfWlkrs / 2, stpIndx, sbstIndx, chn[0].wlkrs, chn[0].rndmVls, chn[0].zRndmVls, chn[0].prpsdWlkrs );
            /* 2 a )  */
            LinearInterpolation <<< Blocks ( nmbrOfWlkrs / 2 ), THRDSPERBLCK >>> ( nmbrOfWlkrs / 2, mdl[0].nmbrOfDistBins, 4, mdl[0].Dist, mdl[0].EBV, mdl[0].errEBV, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh );
            /* 2 ) Assemble array of prior conditions */
            AssembleArrayOfPriors <<< Blocks ( nmbrOfWlkrs / 2 ), THRDSPERBLCK >>> ( nmbrOfWlkrs / 2, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh, chn[0].prrs );
            /* 3 ) Assemble array of absorption factors, spc[0].absrptnFctrs[nmbrOfWlkrs / 2*spc[0].nmbrOfEnrgChnnls] */
            AssembleArrayOfAbsorptionFactors <<< Grid ( spc[0].nmbrOfEnrgChnnls, nmbrOfWlkrs / 2 ), dimBlock >>> ( nmbrOfWlkrs / 2, spc[0].nmbrOfEnrgChnnls, ATNMR, spc[0].crssctns, mdl[0].abndncs, mdl[0].atmcNmbrs, chn[0].prpsdWlkrs, spc[0].absrptnFctrs );
            /* 4 a ) Assemble array of nsa fluxes */
            //BilinearInterpolation <<< dimGrid_1, dimBlock >>> ( nmbrOfWlkrs / 2, spc[0].nmbrOfEnrgChnnls, 2, mdl[0].nsaFlxs, mdl[0].nsaE, mdl[0].nsaT, mdl[0].numNsaE, mdl[0].numNsaT, spc[0].enrgChnnls, chn[0].prpsdWlkrs, spc[0].mdlFlxs );
            /* 4 ) Assemble array of model fluxes, spc[0].mdlFlxs[nmbrOfWlkrs / 2*spc[0].nmbrOfEnrgChnnls] */
            AssembleArrayOfModelFluxes <<< Grid ( spc[0].nmbrOfEnrgChnnls, nmbrOfWlkrs / 2 ), dimBlock >>> ( nmbrOfWlkrs / 2, spc[0].nmbrOfEnrgChnnls, spc[0].enrgChnnls, spc[0].arfFctrs, spc[0].absrptnFctrs, chn[0].prpsdWlkrs, spc[0].mdlFlxs );
            /* 5 ) Fold model fluxes with RMF, spc[0].flddMdlFlxs[nmbrOfWlkrs / 2*spc[0].nmbrOfChnnls] (cuSparse) */
            cdp[0].cusparseStat = cusparseScsrmm ( cdp[0].cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spc[0].nmbrOfChnnls, nmbrOfWlkrs / 2, spc[0].nmbrOfEnrgChnnls, spc[0].nmbrOfRmfVls, &alpha, cdp[0].MatDescr, spc[0].rmfVls, spc[0].rmfPntr, spc[0].rmfIndx, spc[0].mdlFlxs, spc[0].nmbrOfEnrgChnnls, &beta, spc[0].flddMdlFlxs, spc[0].nmbrOfChnnls );
            if ( cdp[0].cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Matrix-matrix multiplication failed yes" ); return stpIndx; }
            /* 6 ) Assemble array of channel statistics, spc[0].chnnlSttstcs[nmbrOfWlkrs / 2*spc[0].nmbrOfChnnls] */
            AssembleArrayOfChannelStatistics <<< Grid ( spc[0].nmbrOfChnnls, nmbrOfWlkrs / 2 ), dimBlock >>> ( nmbrOfWlkrs / 2, spc[0].nmbrOfChnnls, spc[0].srcExptm, spc[0].bckgrndExptm, spc[0].srcCnts, spc[0].bckgrndCnts, spc[0].flddMdlFlxs, spc[0].chnnlSttstcs );
            /* 7 ) Sum up channel statistics, chn[0].prpsdSttstcs[nmbrOfWlkrs / 2] (cuBlas) */
            cdp[0].cublasStat = cublasSgemv ( cdp[0].cublasHandle, CUBLAS_OP_T, spc[0].nmbrOfChnnls, nmbrOfWlkrs / 2, &alpha, spc[0].chnnlSttstcs, spc[0].nmbrOfChnnls, spc[0].ntcdChnnls, incxx, &beta, chn[0].prpsdSttstcs, incyy );
            if ( cdp[0].cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed yes " ); return stpIndx; }
            /* 8 ) Update walkers */
            UpdateWalkers <<< Blocks ( nmbrOfWlkrs / 2 ), THRDSPERBLCK >>> ( nmbrOfWlkrs / 2, stpIndx, sbstIndx, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs, chn[0].prrs, chn[0].zRndmVls, chn[0].rndmVls, chn[0].wlkrs, chn[0].sttstcs );
            /* 9 ) Shift subset index */
            sbstIndx += 1;
        }
        /* Write walkers and statistics to chain,  chnOfWlkrsAndSttstcs[nmbrOfStps*(nmbrOfWlkrs+1)] */
        WriteWalkersAndStatisticsToChain <<< Blocks ( nmbrOfWlkrs ), THRDSPERBLCK >>> ( nmbrOfWlkrs, stpIndx, chn[0].wlkrs, chn[0].sttstcs, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs );
        /* Shift step index */
        stpIndx += 1;
    }

    cudaEventRecord ( cdp[0].stop, 0 );
    cudaEventSynchronize ( cdp[0].stop );

    printf ( "      ... >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done!\n" );

    float elapsedTime;
    cudaEventElapsedTime ( &elapsedTime, cdp[0].start, cdp[0].stop );

    cudaEventRecord ( cdp[0].start, 0 );

    int NN[RANK] = { nmbrOfStps };
    cdp[0].cufftRes = cufftPlanMany ( &cdp[0].cufftPlan, RANK, NN, NULL, 1, nmbrOfStps, NULL, 1, nmbrOfStps, CUFFT_C2C, nmbrOfWlkrs );
    if ( cdp[0].cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error: Direct Plan configuration failed" ); return 1; }
    ReturnChainFunction <<< Grid ( nmbrOfWlkrs, nmbrOfStps ), dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 0, chn[0].chnOfWlkrs, chn[0].chnFnctn );
    AutocorrelationFunctionAveraged ( cdp[0].cufftRes, cdp[0].cublasStat, cdp[0].cublasHandle, cdp[0].cufftPlan, nmbrOfStps, nmbrOfWlkrs, chn[0].chnFnctn, chn[0].atCrrFnctn );

    cudaEventRecord ( cdp[0].stop, 0 );
    cudaEventSynchronize ( cdp[0].stop );

    float cufftElapsedTime;
    cudaEventElapsedTime ( &cufftElapsedTime, cdp[0].start, cdp[0].stop );

    CumulativeSumOfAutocorrelationFunction ( nmbrOfStps, chn[0].atCrrFnctn, chn[0].cmSmAtCrrFnctn );
    int MM = ChooseWindow ( nmbrOfStps, 5e0f, chn[0].cmSmAtCrrFnctn );
    chn[0].atcTime = 2 * chn[0].cmSmAtCrrFnctn[MM] - 1e0f;
    printf ( ".................................................................\n" );
    printf ( " Autocorrelation time window -- %i\n", MM );
    printf ( " Autocorrelation time -- %.8E\n", chn[0].atcTime );
    printf ( " Autocorrelation time threshold -- %.8E\n", nmbrOfStps / 5e1f );
    printf ( " Effective number of independent samples -- %.8E\n", nmbrOfWlkrs * nmbrOfStps / chn[0].atcTime );

    /* Compute and print elapsed time: */
    printf ( ".................................................................\n" );
    printf ( " Time to generate: %3.1f ms\n", elapsedTime );
    printf ( " Time to compute Autocorrelation Function: %3.1f ms\n", cufftElapsedTime );
    printf ( "\n" );

    /* Write results to a file: */
    SimpleWriteDataFloat ( "Autocor.dat", nmbrOfStps, chn[0].atCrrFnctn );
    SimpleWriteDataFloat ( "AutocorCM.dat", nmbrOfStps, chn[0].cmSmAtCrrFnctn );
    WriteChainToFile ( thrdNm, thrdIndx, nmbrOfWlkrs, nmbrOfStps, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs );

    /* Destroy cuda related contexts and things: */
    DestroyAllTheCudaStaff ( cdp[0] );

    /* Free memory: */
    FreeSpec ( spc );
    FreeChain ( chn );
    FreeModel ( mdl );

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cdp[0].err = cudaDeviceReset ( );
    if ( cdp[0].err != cudaSuccess )
    {
        fprintf ( stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString ( cdp[0].err ) );
        exit ( EXIT_FAILURE );
    }

    return 0;
}

#endif // _GWMCMCCUDA_CU_
