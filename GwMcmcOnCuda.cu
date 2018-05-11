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
#include "ReadFitsData.cuh"

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
    printf ( ".................................................................\n" );
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
    const char *thrdNm = argv[3];
    const int nmbrOfWlkrs = atoi ( argv[4] );
    const int nmbrOfHlfTheWlkrs = nmbrOfWlkrs / 2;
    const int nmbrOfStps = atoi ( argv[5] );
    const int thrdIndx = atoi ( argv[6] );
    const int nmbrOfRndmVls = 3 * nmbrOfHlfTheWlkrs * nmbrOfStps;
    const float lwrNtcdEnrg = 0.3;
    const float hghrNtcdEnrg = 8.0;
    const char *abndncsFl = "AngrAbundancesAndRedshift.pars"; // Xset.abund = "angr" and redshift = 0
    const int nmbrOfElmnts = 18; // number of atomic elements
    int atmcNmbrs[nmbrOfElmnts] = { 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 24, 26, 27, 28 };
    int sgFlg = 3; // Xset.xsect = "bcmc"
    const float dlt = 1.E-4;
    const float phbsPwrlwInt[NPRS] = { 1.1, log10f ( 9.E-6 ), 0.1, -3., log10f ( 8E2 ), 0.15 };

//    const char *spLst[] = { "psrj0633.pi", "psrj0633.pi" };
//    int *spDim;
//    int nmbrOfSp = 2;
//    cudaMallocManaged ( ( void ** ) &spDim, nmbrOfSp * sizeof ( int ) );
//    ReadAllTheFitsData ( spLst, spDim );
//    printf ( " Attention! -- %i\n ", spDim[0] );

    /* Allocate array to hold random numbers */
    float *rndmVls;
    cudaMallocManaged ( ( void ** ) &rndmVls, nmbrOfRndmVls * sizeof ( float ) );

    /* Set up initial walkers */
    int prmtrIndx = 0;
    Walker strtngWlkr;
    float *lstWlkrsAndSttstcs;
    cudaMallocManaged ( ( void ** ) &lstWlkrsAndSttstcs, ( NPRS + 1 ) * nmbrOfWlkrs * sizeof ( float ) );
    if ( thrdIndx > 0 )
    {
        ReadLastPositionOfWalkersFromFile ( thrdNm, thrdIndx-1, nmbrOfWlkrs, lstWlkrsAndSttstcs );
    }
    else if ( thrdIndx == 0 )
    {
        strtngWlkr.par[0] = phbsPwrlwInt[0];
        strtngWlkr.par[1] = phbsPwrlwInt[1];
        strtngWlkr.par[2] = phbsPwrlwInt[2];
        strtngWlkr.par[3] = phbsPwrlwInt[3];
        strtngWlkr.par[4] = phbsPwrlwInt[4];
        strtngWlkr.par[NHINDX] = phbsPwrlwInt[NHINDX];
        curandGenerateUniform ( curandGnrtrHst, rndmVls, nmbrOfElmnts - 1 );
        prmtrIndx = NHINDX + 1;
        while ( prmtrIndx < NPRS )
        {
            strtngWlkr.par[prmtrIndx] = dlt * ( 1 - 2 * rndmVls[prmtrIndx-3] );
            prmtrIndx += 1;
        }
        prmtrIndx = 0;
        printf ( " Initial parameters -- " );
        while ( prmtrIndx < NPRS )
        {
            printf ( " %2.2f ", strtngWlkr.par[prmtrIndx] );
            prmtrIndx += 1;
        }
        printf ( "\n" );
        if ( not PriorCondition ( strtngWlkr ) ) { printf ( " !!!Initial walker unsatisfy prior conditions!!!\n" ); }
    }
    printf ( ".................................................................\n" );

    const int nmbrOfDistBins = 442;
    const int numRedCol = 4;

    float *RedData, *Dist, *EBV, *errDist, *errEBV;
    cudaMallocManaged ( ( void ** ) &RedData, nmbrOfDistBins * numRedCol * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &Dist, nmbrOfDistBins * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &EBV, nmbrOfDistBins * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &errDist, nmbrOfDistBins * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &errEBV, nmbrOfDistBins * sizeof ( float ) );

    const char *rddnngFl = "reddening0633.data";
    SimpleReadReddenningData ( rddnngFl, nmbrOfDistBins, RedData, Dist, EBV, errDist, errEBV );

    int numNsaE = 1000;
    int numNsaT = 14;

    float *nsaDt, *nsaE, *nsaT, *nsaFlxs;
    cudaMallocManaged ( ( void ** ) &nsaDt, ( numNsaE + 1 ) * ( numNsaT + 1 ) * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &nsaE, numNsaE * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &nsaT, numNsaT * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &nsaFlxs, numNsaE * numNsaT * sizeof ( float ) );

    const char *nsaFl = "nsa_spec_B_1e12G.dat";
    SimpleReadNsaTable ( nsaFl, numNsaE, numNsaT, nsaDt, nsaT, nsaE, nsaFlxs );

    /* Read abundances and redshift from file: */
    float *abndncs;
    cudaMallocManaged ( ( void ** ) &abndncs, ( NELMS + 1 ) * sizeof ( float ) );
    SimpleReadDataFloat ( abndncsFl, abndncs );
    int *atNmbrs;
    cudaMallocManaged ( ( void ** ) &atNmbrs, nmbrOfElmnts * sizeof ( int ) );
    for ( int i = 0; i < nmbrOfElmnts; i++ )
    {
        atNmbrs[i] = atmcNmbrs[i];
    }

    /* Read FITS information and data: */
    Spectrum spec;
    char srcTbl[FLEN_CARD], arfTbl[FLEN_CARD], rmfTbl[FLEN_CARD], bckgrndTbl[FLEN_CARD];

    ReadFitsInfo ( spcFl, &spec.nmbrOfEnrgChnnls, &spec.nmbrOfChnnls, &spec.nmbrOfRmfVls, &spec.srcExptm, &spec.bckgrndExptm, srcTbl, arfTbl, rmfTbl, bckgrndTbl );

    printf ( " Spectrum table   -- %s\n", srcTbl );
    printf ( " ARF table        -- %s\n", arfTbl );
    printf ( " RMF table        -- %s\n", rmfTbl );
    printf ( " Background table -- %s\n", bckgrndTbl );
    printf ( ".................................................................\n" );
    printf ( " Number of energy channels                = %i\n", spec.nmbrOfEnrgChnnls );
    printf ( " Number of instrument channels            = %i\n", spec.nmbrOfChnnls );
    printf ( " Number of nonzero elements of RMF matrix = %i\n", spec.nmbrOfRmfVls );
    printf ( " Exposure time                            = %.8E\n", spec.srcExptm );
    printf ( " Exposure time (background)               = %.8E\n", spec.bckgrndExptm );


    int *rmfIndxInCsc, *rmfPntr, *rmfIndx;
    cudaMallocManaged ( ( void ** ) &spec.rmfPntrInCsc, ( spec.nmbrOfEnrgChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &rmfIndxInCsc, spec.nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &rmfPntr, ( spec.nmbrOfChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &rmfIndx, spec.nmbrOfRmfVls * sizeof ( int ) );
    float *rmfVlsInCsc, *rmfVls, *enrgChnnls, *arfFctrs, *srcCnts, *bckgrndCnts, *lwrChnnlBndrs, *hghrChnnlBndrs, *gdQltChnnls;
    cudaMallocManaged ( ( void ** ) &rmfVlsInCsc, spec.nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &rmfVls, spec.nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &enrgChnnls, ( spec.nmbrOfEnrgChnnls + 1 ) * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &arfFctrs, spec.nmbrOfEnrgChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &srcCnts, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &bckgrndCnts, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &lwrChnnlBndrs, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &hghrChnnlBndrs, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &gdQltChnnls, spec.nmbrOfChnnls * sizeof ( float ) );

    ReadFitsData ( srcTbl, arfTbl, rmfTbl, bckgrndTbl, spec.nmbrOfEnrgChnnls, spec.nmbrOfChnnls, spec.nmbrOfRmfVls,
                   srcCnts, bckgrndCnts, arfFctrs, rmfVlsInCsc, rmfIndxInCsc, spec.rmfPntrInCsc, gdQltChnnls, lwrChnnlBndrs, hghrChnnlBndrs, enrgChnnls );

    /* Compute absorption crosssections */
    float *crssctns, *absrptnFctrs; //, *absrptnFctrsForUntNhAndFxdAbndncs;
    cudaMallocManaged ( ( void ** ) &crssctns, nmbrOfElmnts * spec.nmbrOfEnrgChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &absrptnFctrs, nmbrOfWlkrs * spec.nmbrOfEnrgChnnls * sizeof ( float ) );

    AssembleArrayOfPhotoelectricCrossections ( spec.nmbrOfEnrgChnnls, nmbrOfElmnts, sgFlg, enrgChnnls, atmcNmbrs, crssctns );

    /* Allocate walkers, spectra etc.: */
    float *prrs;
    cudaMallocManaged ( ( void ** ) &prrs, nmbrOfHlfTheWlkrs * sizeof ( float ) );
    Walker *wlkrs, *prpsdWlkrs, *chnOfWlkrs;
    cudaMallocManaged ( ( void ** ) &wlkrs, nmbrOfWlkrs * sizeof ( Walker ) );
    cudaMallocManaged ( ( void ** ) &prpsdWlkrs, nmbrOfHlfTheWlkrs * sizeof ( Walker ) );
    cudaMallocManaged ( ( void ** ) &chnOfWlkrs, nmbrOfWlkrs * nmbrOfStps * sizeof ( Walker ) );
    float *mdlFlxs, *flddMdlFlxs, *ntcdChnnls, *chnnlSttstcs, *sttstcs, *prpsdSttstcs, *chnOfSttstcs, *zRndmVls;
    cudaMallocManaged ( ( void ** ) &mdlFlxs, nmbrOfWlkrs * spec.nmbrOfEnrgChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &flddMdlFlxs, nmbrOfWlkrs * spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &ntcdChnnls, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &chnnlSttstcs, spec.nmbrOfChnnls * nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &sttstcs, nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &prpsdSttstcs, nmbrOfHlfTheWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &chnOfSttstcs, nmbrOfWlkrs * nmbrOfStps * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &zRndmVls, nmbrOfHlfTheWlkrs * sizeof ( float ) );
    float *mNh, *sNh;
    cudaMallocManaged ( ( void ** ) &mNh, nmbrOfHlfTheWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &sNh, nmbrOfHlfTheWlkrs * sizeof ( float ) );

    /* Set auxiliary parameters, threads, blocks etc.:  */
    int incxx = INCXX, incyy = INCYY;
    float alpha = ALPHA, beta = BETA;
    int thrdsPerBlck = THRDSPERBLCK;
    dim3 dimBlock ( thrdsPerBlck, thrdsPerBlck );
    int blcksPerThrd_0 = ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck;
    int blcksPerThrd_1 = ( nmbrOfHlfTheWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck;
    int blcksPerThrd_2 = ( spec.nmbrOfChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck;

    dim3 dimGrid_0 ( ( spec.nmbrOfEnrgChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_1 ( ( spec.nmbrOfEnrgChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfHlfTheWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_2 ( ( spec.nmbrOfChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_3 ( ( spec.nmbrOfChnnls + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfHlfTheWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_4 ( ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck );

    /* Transpose RMF matrix: */
    cusparseStat = cusparseScsr2csc ( cusparseHandle, spec.nmbrOfEnrgChnnls, spec.nmbrOfChnnls, spec.nmbrOfRmfVls, rmfVlsInCsc, spec.rmfPntrInCsc, rmfIndxInCsc,
                       rmfVls, rmfIndx, rmfPntr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO );
    if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: RMF transpose failed " ); return 1; }
    /* Assemble array of noticed channels, ntcdChnnls[spec.nmbrOfChnnls] */
    AssembleArrayOfNoticedChannels <<< blcksPerThrd_2, thrdsPerBlck >>> ( spec.nmbrOfChnnls, lwrNtcdEnrg, hghrNtcdEnrg, lwrChnnlBndrs, hghrChnnlBndrs, gdQltChnnls, ntcdChnnls );
    /* Calculate number of noticed channels */
    float smOfNtcdChnnls;
    cublasStat = cublasSdot ( cublasHandle, spec.nmbrOfChnnls, ntcdChnnls, incxx, ntcdChnnls, incyy, &smOfNtcdChnnls );
    if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: channel summation failed " ); return 1; }
    cudaDeviceSynchronize ( );
    printf ( ".................................................................\n" );
    printf ( " Number of used instrument channels -- %4.0f\n", smOfNtcdChnnls );
    printf ( " Number of degrees of freedom -- %4.0f\n", smOfNtcdChnnls - NPRS );
    printf ( ".................................................................\n" );

    if ( thrdIndx > 0 )
    {
        /* Initialize walkers and statistics from last chain */
        InitializeWalkersAndStatisticsFromLastChain <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfWlkrs, lstWlkrsAndSttstcs, wlkrs, sttstcs );
    }
    else if ( thrdIndx == 0 )
    {
        /* 1 ) Generate uniformly distributed floating point values between 0.0 and 1.0, rndmVls[nmbrOfWlkrs] (cuRand) */
        curandGenerateUniform ( curandGnrtr, rndmVls, nmbrOfWlkrs );
        /* 2 ) Initialize walkers, actlWlkrs[nmbrOfWlkrs] */
        InitializeWalkersAtRandom <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfWlkrs, dlt, strtngWlkr, rndmVls, wlkrs );
        /* 3 ) Assemble array of absorption factors, absrptnFctrs[nmbrOfWlkrs*spec.nmbrOfEnrgChnnls] */
        AssembleArrayOfAbsorptionFactors <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, nmbrOfElmnts, crssctns, abndncs, atNmbrs, wlkrs, absrptnFctrs );
        /* 4 a ) Assemble array of nsa fluxes */
        //BilinearInterpolation <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, 2, nsaFlxs, nsaE, nsaT, numNsaE, numNsaT, enrgChnnls, wlkrs, mdlFlxs );
        /* 4 ) Assemble array of model fluxes, mdlFlxs[nmbrOfWlkrs*spec.nmbrOfEnrgChnnls] */
        AssembleArrayOfModelFluxes <<< dimGrid_0, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, enrgChnnls, arfFctrs, absrptnFctrs, wlkrs, mdlFlxs );
        /* 5 ) Fold model fluxes with RMF, flddMdlFlxs[nmbrOfWlkrs*spec.nmbrOfChnnls] (cuSparse) */
        cusparseStat = cusparseScsrmm ( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spec.nmbrOfChnnls, nmbrOfWlkrs, spec.nmbrOfEnrgChnnls, spec.nmbrOfRmfVls, &alpha, MatDescr,
                                        rmfVls, rmfPntr, rmfIndx, mdlFlxs, spec.nmbrOfEnrgChnnls, &beta, flddMdlFlxs, spec.nmbrOfChnnls );
        if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Matrix-matrix multiplication failed yes " ); return 1; }
        /* 6 ) Assemble array of channel statistics, chnnlSttstcs[nmbrOfWlkrs*spec.nmbrOfChnnls] */
        AssembleArrayOfChannelStatistics <<< dimGrid_2, dimBlock >>> ( nmbrOfWlkrs, spec.nmbrOfChnnls, spec.srcExptm, spec.bckgrndExptm, srcCnts, bckgrndCnts, flddMdlFlxs, chnnlSttstcs );
        /* 7 ) Sum up channel statistics, actlSttstcs[nmbrOfWlkrs] (cuBlas) */
        cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, spec.nmbrOfChnnls, nmbrOfWlkrs, &alpha, chnnlSttstcs, spec.nmbrOfChnnls, ntcdChnnls, incxx, &beta, sttstcs, incyy );
        if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed 0 " ); return 1; }
    }

    printf ( " Start ...                                                  \n" );

    cudaEvent_t start, stop;
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    cudaEventRecord ( start, 0 );

    /* Generate uniformly distributed floating point values between 0.0 and 1.0, rndmVls[3*nmbrOfHlfTheWlkrs*nmbrOfStps] (cuRand) */
    curandGenerateUniform ( curandGnrtr, rndmVls, 3 * nmbrOfHlfTheWlkrs * nmbrOfStps );

    /* Start MCMC !!!!! */
    int stpIndx = 0, sbstIndx;
    while ( stpIndx < nmbrOfStps )
    {
        /* Initialize subset index */
        sbstIndx = 0;
        /* Iterate over two subsets */
        while ( sbstIndx < 2 )
        {
            /* 1 ) Generate Z values and proposed walkers, zRndmVls[nmbrOfHlfTheWlkrs], prpsdWlkrs[nmbrOfHlfTheWlkrs] */
            GenerateProposal <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, stpIndx, sbstIndx, wlkrs, rndmVls, zRndmVls, prpsdWlkrs );
            /* 2 a )  */
            LinearInterpolation <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, nmbrOfDistBins, 4, Dist, EBV, errEBV, prpsdWlkrs, mNh, sNh );
            /* 2 ) Assemble array of prior conditions */
            AssembleArrayOfPriors <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, prpsdWlkrs, mNh, sNh, prrs );
            /* 3 ) Assemble array of absorption factors, absrptnFctrs[nmbrOfHlfTheWlkrs*spec.nmbrOfEnrgChnnls] */
            AssembleArrayOfAbsorptionFactors <<< dimGrid_1, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec.nmbrOfEnrgChnnls, nmbrOfElmnts, crssctns, abndncs, atNmbrs, prpsdWlkrs, absrptnFctrs );
            /* 4 a ) Assemble array of nsa fluxes */
            //BilinearInterpolation <<< dimGrid_1, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec.nmbrOfEnrgChnnls, 2, nsaFlxs, nsaE, nsaT, numNsaE, numNsaT, enrgChnnls, prpsdWlkrs, mdlFlxs );
            /* 4 ) Assemble array of model fluxes, mdlFlxs[nmbrOfHlfTheWlkrs*spec.nmbrOfEnrgChnnls] */
            AssembleArrayOfModelFluxes <<< dimGrid_1, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec.nmbrOfEnrgChnnls, enrgChnnls, arfFctrs, absrptnFctrs, prpsdWlkrs, mdlFlxs );
            /* 5 ) Fold model fluxes with RMF, flddMdlFlxs[nmbrOfHlfTheWlkrs*spec.nmbrOfChnnls] (cuSparse) */
            cusparseStat = cusparseScsrmm ( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, spec.nmbrOfChnnls, nmbrOfHlfTheWlkrs, spec.nmbrOfEnrgChnnls, spec.nmbrOfRmfVls, &alpha, MatDescr,
                                            rmfVls, rmfPntr, rmfIndx, mdlFlxs, spec.nmbrOfEnrgChnnls, &beta, flddMdlFlxs, spec.nmbrOfChnnls );
            if ( cusparseStat != CUSPARSE_STATUS_SUCCESS ) { fprintf ( stderr, " CUSPARSE error: Matrix-matrix multiplication failed yes" ); return stpIndx; }
            /* 6 ) Assemble array of channel statistics, chnnlSttstcs[nmbrOfHlfTheWlkrs*spec.nmbrOfChnnls] */
            AssembleArrayOfChannelStatistics <<< dimGrid_3, dimBlock >>> ( nmbrOfHlfTheWlkrs, spec.nmbrOfChnnls, spec.srcExptm, spec.bckgrndExptm, srcCnts, bckgrndCnts, flddMdlFlxs, chnnlSttstcs );
            /* 7 ) Sum up channel statistics, prpsdSttstcs[nmbrOfHlfTheWlkrs] (cuBlas) */
            cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, spec.nmbrOfChnnls, nmbrOfHlfTheWlkrs, &alpha, chnnlSttstcs, spec.nmbrOfChnnls, ntcdChnnls, incxx, &beta, prpsdSttstcs, incyy );
            if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: Matrix-vector multiplication failed yes " ); return stpIndx; }
            /* 8 ) Update walkers */
            UpdateWalkers <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfHlfTheWlkrs, stpIndx, sbstIndx, prpsdWlkrs, prpsdSttstcs, prrs, zRndmVls, rndmVls, wlkrs, sttstcs );
            /* 9 ) Shift subset index */
            sbstIndx += 1;
        }
        /* Write walkers and statistics to chain,  chnOfWlkrsAndSttstcs[nmbrOfStps*(nmbrOfWlkrs+1)] */
        WriteWalkersAndStatisticsToChain <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfWlkrs, stpIndx, wlkrs, sttstcs, chnOfWlkrs, chnOfSttstcs );
        /* Shift step index */
        stpIndx += 1;
    }

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );

    printf ( "      ... >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done!\n" );
    printf ( ".................................................................\n" );

    float elapsedTime;
    cudaEventElapsedTime ( &elapsedTime, start, stop );

    float *chnFnctn, *atCrrFnctn, *cmSmAtCrrFnctn;
    cudaMallocManaged ( ( void ** ) &chnFnctn, nmbrOfStps * nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &atCrrFnctn, nmbrOfStps * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &cmSmAtCrrFnctn, nmbrOfStps * sizeof ( float ) );

    int NN[RANK] = { nmbrOfStps };
    cufftRes = cufftPlanMany ( &cufftPlan, RANK, NN, NULL, 1, nmbrOfStps, NULL, 1, nmbrOfStps, CUFFT_C2C, nmbrOfWlkrs );
    if ( cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error: Direct Plan configuration failed" ); return 1; }

    cudaEventRecord ( start, 0 );

    ReturnChainFunction <<< dimGrid_4, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 0, chnOfWlkrs, chnFnctn );
    AutocorrelationFunctionAveraged ( cufftRes, cublasStat, cublasHandle, cufftPlan, nmbrOfStps, nmbrOfWlkrs, chnFnctn, atCrrFnctn );

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );

    float cufftElapsedTime;
    cudaEventElapsedTime ( &cufftElapsedTime, start, stop );

    CumulativeSumOfAutocorrelationFunction ( nmbrOfStps, atCrrFnctn, cmSmAtCrrFnctn );
    int MM = ChooseWindow ( nmbrOfStps, 5e0f, cmSmAtCrrFnctn );

    float atcTime;
    atcTime = 2 * cmSmAtCrrFnctn[MM] - 1e0f;

    printf ( " Autocorrelation time window -- %i\n", MM );
    printf ( " Autocorrelation time -- %.8E\n", atcTime );
    printf ( " Autocorrelation time threshold -- %.8E\n", nmbrOfStps / 5e1f );
    printf ( " Effective number of independent samples -- %.8E\n", nmbrOfWlkrs * nmbrOfStps / atcTime );
    printf ( ".................................................................\n" );

    /* Compute and print elapsed time: */
    printf ( " Time to generate: %3.1f ms\n", elapsedTime );
    printf ( " Time to compute Autocorrelation Function: %3.1f ms\n", cufftElapsedTime );
    printf ( "\n" );

    /* Write results to a file: */
    SimpleWriteDataFloat ( "Autocor.dat", nmbrOfStps, atCrrFnctn );
    SimpleWriteDataFloat ( "AutocorCM.dat", nmbrOfStps, cmSmAtCrrFnctn );
    WriteChainToFile ( thrdNm, thrdIndx, nmbrOfWlkrs, nmbrOfStps, chnOfWlkrs, chnOfSttstcs );

    /* Destroy cuda related contexts and things: */
    cusparseDestroy ( cusparseHandle );
    cublasDestroy ( cublasHandle );
    curandDestroyGenerator ( curandGnrtr );
    curandDestroyGenerator ( curandGnrtrHst );
    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );
    cufftDestroy ( cufftPlan );

    /* Free memory: */
    cudaFree ( chnFnctn );
    cudaFree ( atCrrFnctn );
    cudaFree ( cmSmAtCrrFnctn );
    cudaFree ( rndmVls );
    cudaFree ( lstWlkrsAndSttstcs );
    cudaFree ( abndncs );
    cudaFree ( crssctns );
    cudaFree ( absrptnFctrs );
    cudaFree ( rmfVlsInCsc );
    cudaFree ( rmfIndxInCsc );
    cudaFree ( spec.rmfPntrInCsc );
    cudaFree ( rmfVls );
    cudaFree ( rmfIndx );
    cudaFree ( rmfPntr );
    cudaFree ( enrgChnnls );
    cudaFree ( arfFctrs );
    cudaFree ( srcCnts );
    cudaFree ( bckgrndCnts );
    cudaFree ( gdQltChnnls );
    cudaFree ( lwrChnnlBndrs );
    cudaFree ( hghrChnnlBndrs );
    cudaFree ( mdlFlxs );
    cudaFree ( flddMdlFlxs );
    cudaFree ( sttstcs );
    cudaFree ( prpsdSttstcs );
    cudaFree ( chnnlSttstcs );
    cudaFree ( wlkrs );
    cudaFree ( prpsdWlkrs );
    cudaFree ( zRndmVls );
    cudaFree ( prrs );
    cudaFree ( chnOfWlkrs );
    cudaFree ( chnOfSttstcs );
    cudaFree ( ntcdChnnls );
    cudaFree ( mNh );
    cudaFree ( sNh );
    cudaFree ( RedData );
    cudaFree ( Dist );
    cudaFree ( EBV );
    cudaFree ( errDist );
    cudaFree ( errEBV );
    cudaFree ( nsaDt );
    cudaFree ( nsaT );
    cudaFree ( nsaE );
    cudaFree ( nsaFlxs );
    cudaFree ( atNmbrs );
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
