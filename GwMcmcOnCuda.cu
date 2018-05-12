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
    const int verbose = 1;
    /* cuda succes status: */
    cudaError_t err = cudaSuccess;
    /* cuda runtime version */
    int runtimeVersion[4], driverVersion[4];
    cudaRuntimeGetVersion ( runtimeVersion );
    cudaDriverGetVersion ( driverVersion );
    printf ( "\n" );
    printf ( ".................................................................\n" );
    printf ( " Driver API: v%d \n", driverVersion[0] );
    printf ( " Runtime API: v%d \n", runtimeVersion[0] );
    /* Set and enquire about cuda device */
    cudaDeviceProp prop;
    int dev;
    cudaSetDevice ( atoi( argv[1] ) );
    cudaGetDevice ( &dev );
    printf ( " CUDA device ID: %d\n", dev );
    cudaGetDeviceProperties ( &prop, dev );
    printf ( " CUDA device Name: %s\n", prop.name );
    /* cuSparse related things */
    cusparseStatus_t cusparseStat;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t MatDescr = 0;
    cusparseStat = cusparseCreate ( &cusparseHandle );
    if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Creation of cuSparse context failed " ); return 1; }
    cusparseStat = cusparseCreateMatDescr ( &MatDescr );
    if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Creation of matrix descriptor failed " ); return 1; }
    cusparseStat = cusparseSetMatType ( MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
    if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Setting matrix type to general failed " ); return 1; }
    cusparseStat = cusparseSetMatIndexBase ( MatDescr, CUSPARSE_INDEX_BASE_ZERO );
    if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Setting to base zero index failed " ); return 1; }
    /* cuBlas related things */
    cublasStatus_t cublasStat;
    cublasHandle_t cublasHandle = 0;
    cublasStat = cublasCreate ( &cublasHandle );
    if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Creation of cuBlas context failed " ); return 1; }
    /* cuRand related things */
    curandGenerator_t curandGnrtr, curandGnrtrHst;
    curandCreateGenerator ( &curandGnrtr, CURAND_RNG_PSEUDO_DEFAULT );
    curandCreateGeneratorHost ( &curandGnrtrHst, CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed ( curandGnrtr, 1234ULL );
    curandSetPseudoRandomGeneratorSeed ( curandGnrtrHst, 1234ULL );
    /* cuFfft related things */
    cufftResult_t cufftRes;
    cufftHandle cufftPlan;

    /* Set up initial parameters: */
    const char *spcFl = argv[2];
    const char *spcLst[NSPCTR] = { spcFl, spcFl };
    const char *thrdNm = argv[3];
    const int nmbrOfWlkrs = atoi ( argv[4] );
    const int nmbrOfHlfTheWlkrs = nmbrOfWlkrs / 2;
    const int nmbrOfStps = atoi ( argv[5] );
    const int thrdIndx = atoi ( argv[6] );
    const float lwrNtcdEnrg = 0.3;
    const float hghrNtcdEnrg = 8.0;
    const char *abndncsFl = "AngrAbundancesAndRedshift.pars"; // Xset.abund = "angr" and redshift = 0
    const int atNm[ATNMR] = { 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 24, 26, 27, 28 };
    const char *rddnngFl = "reddening0633.data";
    const int nmbrOfDistBins = 442;
    const int numRedCol = 4;
    const char *nsaFl = "nsa_spec_B_1e12G.dat";
    int numNsaE = 1000;
    int numNsaT = 14;
    int sgFlg = 3; // Xset.xsect = "bcmc"
    const float dlt = 1.E-4;
    const float phbsPwrlwInt[NPRS] = { 1.1, log10f ( 9.E-6 ), 0.1, -3., log10f ( 8E2 ), 0.15 };

    int *atmcNmbrs;
    cudaMallocManaged ( ( void ** ) &atmcNmbrs, ATNMR * sizeof ( int ) );
    for ( int i = 0; i < ATNMR; i++ ) { atmcNmbrs[i] = atNm[i]; }

    /* Read abundances and redshift from file: */
    float *abndncs;
    cudaMallocManaged ( ( void ** ) &abndncs, ( NELMS + 1 ) * sizeof ( float ) );
    SimpleReadDataFloat ( abndncsFl, abndncs );

    /* Read reddening data */
    float *RedData, *Dist, *EBV, *errDist, *errEBV;
    cudaMallocManaged ( ( void ** ) &RedData, nmbrOfDistBins * numRedCol * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &Dist, nmbrOfDistBins * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &EBV, nmbrOfDistBins * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &errDist, nmbrOfDistBins * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &errEBV, nmbrOfDistBins * sizeof ( float ) );
    SimpleReadReddenningData ( rddnngFl, nmbrOfDistBins, RedData, Dist, EBV, errDist, errEBV );

    /* Read NSA data */
    float *nsaDt, *nsaE, *nsaT, *nsaFlxs;
    cudaMallocManaged ( ( void ** ) &nsaDt, ( numNsaE + 1 ) * ( numNsaT + 1 ) * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &nsaE, numNsaE * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &nsaT, numNsaT * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &nsaFlxs, numNsaE * numNsaT * sizeof ( float ) );
    SimpleReadNsaTable ( nsaFl, numNsaE, numNsaT, nsaDt, nsaT, nsaE, nsaFlxs );

    /* Allocate walkers etc.: */
    Chain chn[NSPCTR];
    InitializeChain ( thrdNm, thrdIndx, nmbrOfWlkrs, nmbrOfStps, phbsPwrlwInt, curandGnrtrHst, dlt, chn );

    /* Read FITS data and allocated spectra and model spectra arrays */
    Spectrum spec[NSPCTR];
    InitializeSpectra ( spcLst, verbose, nmbrOfWlkrs, spec );

    /* Set auxiliary parameters, threads, blocks etc.:  */
    int incxx = INCXX, incyy = INCYY;
    float alpha = ALPHA, beta = BETA;
    int thrdsPerBlck = THRDSPERBLCK;
    dim3 dimBlock ( thrdsPerBlck, thrdsPerBlck );
    int blcksPerThrd_0 = ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck;
    int blcksPerThrd_1 = ( nmbrOfHlfTheWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck;
    int blcksPerThrd_2 = ( spec[0].nmbrOfChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck;

    dim3 dimGrid_0 ( ( spec[0].nmbrOfEnrgChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_1 ( ( spec[0].nmbrOfEnrgChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfHlfTheWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_2 ( ( spec[0].nmbrOfChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_3 ( ( spec[0].nmbrOfChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfHlfTheWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_4 ( ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck );

    /* Transpose RMF matrix: */
    cusparseStat = cusparseScsr2csc ( cusparseHandle, spec[0].nmbrOfEnrgChnnls, spec[0].nmbrOfChnnls, spec[0].nmbrOfRmfVls, spec[0].rmfVlsInCsc, spec[0].rmfPntrInCsc, spec[0].rmfIndxInCsc, spec[0].rmfVls, spec[0].rmfIndx, spec[0].rmfPntr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO );
    if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: RMF transpose failed " ); return 1; }
    /* Assemble array of noticed channels, spec[0].ntcdChnnls[spec[0].nmbrOfChnnls] */
    AssembleArrayOfNoticedChannels <<< blcksPerThrd_2, thrdsPerBlck >>> ( spec[0].nmbrOfChnnls, lwrNtcdEnrg, hghrNtcdEnrg, spec[0].lwrChnnlBndrs, spec[0].hghrChnnlBndrs, spec[0].gdQltChnnls, spec[0].ntcdChnnls );
    /* Calculate number of noticed channels */
    float smOfNtcdChnnls;
    cublasStat = cublasSdot ( cublasHandle, spec[0].nmbrOfChnnls, spec[0].ntcdChnnls, incxx, spec[0].ntcdChnnls, incyy, &smOfNtcdChnnls );
    if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: channel summation failed " ); return 1; }
    cudaDeviceSynchronize ( );
    printf ( ".................................................................\n" );
    printf ( " Number of used instrument channels -- %4.0f\n", smOfNtcdChnnls );
    printf ( " Number of degrees of freedom -- %4.0f\n", smOfNtcdChnnls - NPRS );

    /* Compute absorption crosssections */
    AssembleArrayOfPhotoelectricCrossections ( spec[0].nmbrOfEnrgChnnls, ATNMR, sgFlg, spec[0].enrgChnnls, atmcNmbrs, spec[0].crssctns );

    if ( thrdIndx > 0 )
    {
        /* Initialize walkers and statistics from last chain */
        InitializeWalkersAndStatisticsFromLastChain <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfWlkrs, chn[0].lstWlkrsAndSttstcs, chn[0].wlkrs, chn[0].sttstcs );
    }
    else if ( thrdIndx == 0 )
    {
        /* 1 ) Generate uniformly distributed floating point values between 0.0 and 1.0, chn[0].rndmVls[nmbrOfWlkrs] (cuRand) */
        curandGenerateUniform ( curandGnrtr, chn[0].rndmVls, nmbrOfWlkrs );
        /* 2 ) Initialize walkers, actlWlkrs[nmbrOfWlkrs] */
        InitializeWalkersAtRandom <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfWlkrs, dlt, chn[0].strtngWlkr, chn[0].rndmVls, chn[0].wlkrs );
        /* 3 ) Assemble array of absorption factors, spec[0].absrptnFctrs[nmbrOfWlkrs*spec[0].nmbrOfEnrgChnnls] */
        AssembleArrayOfAbsorptionFactors <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spec[0].nmbrOfEnrgChnnls, ATNMR, spec[0].crssctns, abndncs, atmcNmbrs, chn[0].wlkrs, spec[0].absrptnFctrs );
        /* 4 a ) Assemble array of nsa fluxes */
        //BilinearInterpolation <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spec[0].nmbrOfEnrgChnnls, 2, nsaFlxs, nsaE, nsaT, numNsaE, numNsaT, spec[0].enrgChnnls, chn[0].wlkrs, spec[0].mdlFlxs );
        /* 4 ) Assemble array of model fluxes, spec[0].mdlFlxs[nmbrOfWlkrs*spec[0].nmbrOfEnrgChnnls] */
        AssembleArrayOfModelFluxes <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spec[0].nmbrOfEnrgChnnls, spec[0].enrgChnnls, spec[0].arfFctrs, spec[0].absrptnFctrs, chn[0].wlkrs, spec[0].mdlFlxs );
        /* 5 ) Fold model fluxes with RMF, spec[0].flddMdlFlxs[nmbrOfWlkrs*spec[0].nmbrOfChnnls] (cuSparse) */
        cusparseStat = cusparseScsrmm ( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spec[0].nmbrOfChnnls, nmbrOfWlkrs, spec[0].nmbrOfEnrgChnnls, spec[0].nmbrOfRmfVls, &alpha, MatDescr,
                                        spec[0].rmfVls, spec[0].rmfPntr, spec[0].rmfIndx, spec[0].mdlFlxs, spec[0].nmbrOfEnrgChnnls, &beta, spec[0].flddMdlFlxs, spec[0].nmbrOfChnnls );
        if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Matrix-matrix multiplication failed yes " ); return 1; }
        /* 6 ) Assemble array of channel statistics, spec[0].chnnlSttstcs[nmbrOfWlkrs*spec[0].nmbrOfChnnls] */
        AssembleArrayOfChannelStatistics <<< dimGrid_2, dimBlock >>> ( nmbrOfWlkrs, spec[0].nmbrOfChnnls, spec[0].srcExptm, spec[0].bckgrndExptm, spec[0].srcCnts, spec[0].bckgrndCnts, spec[0].flddMdlFlxs, spec[0].chnnlSttstcs );
        /* 7 ) Sum up channel statistics, actlSttstcs[nmbrOfWlkrs] (cuBlas) */
        cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, spec[0].nmbrOfChnnls, nmbrOfWlkrs, &alpha, spec[0].chnnlSttstcs, spec[0].nmbrOfChnnls, spec[0].ntcdChnnls, incxx, &beta, chn[0].sttstcs, incyy );
        if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed 0 " ); return 1; }
    }

    printf ( ".................................................................\n" );
    printf ( " Start ...                                                  \n" );

    cudaEvent_t start, stop;
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    cudaEventRecord ( start, 0 );

    /* Generate uniformly distributed floating point values between 0.0 and 1.0, chn[0].rndmVls[3*nmbrOfHlfTheWlkrs*nmbrOfStps] (cuRand) */
    curandGenerateUniform ( curandGnrtr, chn[0].rndmVls, 3 * nmbrOfHlfTheWlkrs * nmbrOfStps );

    /* Start MCMC !!!!! */
    int stpIndx = 0, sbstIndx;
    while ( stpIndx < nmbrOfStps )
    {
        /* Initialize subset index */
        sbstIndx = 0;
        /* Iterate over two subsets */
        while ( sbstIndx < 2 )
        {
            /* 1 ) Generate Z values and proposed walkers, chn[0].zRndmVls[nmbrOfHlfTheWlkrs], chn[0].prpsdWlkrs[nmbrOfHlfTheWlkrs] */
            GenerateProposal <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, stpIndx, sbstIndx, chn[0].wlkrs, chn[0].rndmVls, chn[0].zRndmVls, chn[0].prpsdWlkrs );
            /* 2 a )  */
            LinearInterpolation <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, nmbrOfDistBins, 4, Dist, EBV, errEBV, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh );
            /* 2 ) Assemble array of prior conditions */
            AssembleArrayOfPriors <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, chn[0].prpsdWlkrs, chn[0].mNh, chn[0].sNh, chn[0].prrs );
            /* 3 ) Assemble array of absorption factors, spec[0].absrptnFctrs[nmbrOfHlfTheWlkrs*spec[0].nmbrOfEnrgChnnls] */
            AssembleArrayOfAbsorptionFactors <<< dimGrid_1, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec[0].nmbrOfEnrgChnnls, ATNMR, spec[0].crssctns, abndncs, atmcNmbrs, chn[0].prpsdWlkrs, spec[0].absrptnFctrs );
            /* 4 a ) Assemble array of nsa fluxes */
            //BilinearInterpolation <<< dimGrid_1, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec[0].nmbrOfEnrgChnnls, 2, nsaFlxs, nsaE, nsaT, numNsaE, numNsaT, spec[0].enrgChnnls, chn[0].prpsdWlkrs, spec[0].mdlFlxs );
            /* 4 ) Assemble array of model fluxes, spec[0].mdlFlxs[nmbrOfHlfTheWlkrs*spec[0].nmbrOfEnrgChnnls] */
            AssembleArrayOfModelFluxes <<< dimGrid_1, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec[0].nmbrOfEnrgChnnls, spec[0].enrgChnnls, spec[0].arfFctrs, spec[0].absrptnFctrs, chn[0].prpsdWlkrs, spec[0].mdlFlxs );
            /* 5 ) Fold model fluxes with RMF, spec[0].flddMdlFlxs[nmbrOfHlfTheWlkrs*spec[0].nmbrOfChnnls] (cuSparse) */
            cusparseStat = cusparseScsrmm ( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spec[0].nmbrOfChnnls, nmbrOfHlfTheWlkrs, spec[0].nmbrOfEnrgChnnls, spec[0].nmbrOfRmfVls, &alpha, MatDescr,
                                            spec[0].rmfVls, spec[0].rmfPntr, spec[0].rmfIndx, spec[0].mdlFlxs, spec[0].nmbrOfEnrgChnnls, &beta, spec[0].flddMdlFlxs, spec[0].nmbrOfChnnls );
            if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Matrix-matrix multiplication failed yes" ); return stpIndx; }
            /* 6 ) Assemble array of channel statistics, spec[0].chnnlSttstcs[nmbrOfHlfTheWlkrs*spec[0].nmbrOfChnnls] */
            AssembleArrayOfChannelStatistics <<< dimGrid_3, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec[0].nmbrOfChnnls, spec[0].srcExptm, spec[0].bckgrndExptm, spec[0].srcCnts, spec[0].bckgrndCnts, spec[0].flddMdlFlxs, spec[0].chnnlSttstcs );
            /* 7 ) Sum up channel statistics, chn[0].prpsdSttstcs[nmbrOfHlfTheWlkrs] (cuBlas) */
            cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, spec[0].nmbrOfChnnls, nmbrOfHlfTheWlkrs, &alpha, spec[0].chnnlSttstcs, spec[0].nmbrOfChnnls, spec[0].ntcdChnnls, incxx, &beta, chn[0].prpsdSttstcs, incyy );
            if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed yes " ); return stpIndx; }
            /* 8 ) Update walkers */
            UpdateWalkers <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, stpIndx, sbstIndx, chn[0].prpsdWlkrs, chn[0].prpsdSttstcs, chn[0].prrs, chn[0].zRndmVls, chn[0].rndmVls, chn[0].wlkrs, chn[0].sttstcs );
            /* 9 ) Shift subset index */
            sbstIndx += 1;
        }
        /* Write walkers and statistics to chain,  chnOfWlkrsAndSttstcs[nmbrOfStps*(nmbrOfWlkrs+1)] */
        WriteWalkersAndStatisticsToChain <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfWlkrs, stpIndx, chn[0].wlkrs, chn[0].sttstcs, chn[0].chnOfWlkrs, chn[0].chnOfSttstcs );
        /* Shift step index */
        stpIndx += 1;
    }

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );

    printf ( "      ... >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done!\n" );

    float elapsedTime;
    cudaEventElapsedTime ( &elapsedTime, start, stop );

    cudaEventRecord ( start, 0 );

    int NN[RANK] = { nmbrOfStps };
    cufftRes = cufftPlanMany ( &cufftPlan, RANK, NN, NULL, 1, nmbrOfStps, NULL, 1, nmbrOfStps, CUFFT_C2C, nmbrOfWlkrs );
    if ( cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error: Direct Plan configuration failed" ); return 1; }
    ReturnChainFunction <<< dimGrid_4, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 0, chn[0].chnOfWlkrs, chn[0].chnFnctn );
    AutocorrelationFunctionAveraged ( cufftRes, cublasStat, cublasHandle, cufftPlan, nmbrOfStps, nmbrOfWlkrs, chn[0].chnFnctn, chn[0].atCrrFnctn );

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );

    float cufftElapsedTime;
    cudaEventElapsedTime ( &cufftElapsedTime, start, stop );

    CumulativeSumOfAutocorrelationFunction ( nmbrOfStps, chn[0].atCrrFnctn, chn[0].cmSmAtCrrFnctn );
    int MM = ChooseWindow ( nmbrOfStps, 5e0f, chn[0].cmSmAtCrrFnctn );
    float atcTime;
    atcTime = 2 * chn[0].cmSmAtCrrFnctn[MM] - 1e0f;
    printf ( ".................................................................\n" );
    printf ( " Autocorrelation time window -- %i\n", MM );
    printf ( " Autocorrelation time -- %.8E\n", atcTime );
    printf ( " Autocorrelation time threshold -- %.8E\n", nmbrOfStps / 5e1f );
    printf ( " Effective number of independent samples -- %.8E\n", nmbrOfWlkrs * nmbrOfStps / atcTime );

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
    cusparseDestroy ( cusparseHandle );
    cublasDestroy ( cublasHandle );
    curandDestroyGenerator ( curandGnrtr );
    curandDestroyGenerator ( curandGnrtrHst );
    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );
    cufftDestroy ( cufftPlan );

    /* Free memory: */
    FreeSpec ( spec );
    FreeChain ( chn );

    cudaFree ( abndncs );
    cudaFree ( RedData );
    cudaFree ( Dist );
    cudaFree ( EBV );
    cudaFree ( errDist );
    cudaFree ( errEBV );
    cudaFree ( nsaDt );
    cudaFree ( nsaT );
    cudaFree ( nsaE );
    cudaFree ( nsaFlxs );
    cudaFree ( atmcNmbrs );
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset ( );
    if ( err != cudaSuccess )
    {
        fprintf ( stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString ( err ) );
        exit ( EXIT_FAILURE );
    }

    return 0;
}

#endif // _GWMCMCCUDA_CU_
