#ifndef _GWMCMCSTRUCTURESFUNCTIONSANDKERNELS_CU_
#define _GWMCMCSTRUCTURESFUNCTIONSANDKERNELS_CU_

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

/* Functions: */
__host__ __device__ Walker AddWalkers ( Walker a, Walker b )
{
    Walker c;
    for ( int i = 0; i < NPRS; i++ )
    {
        c.par[i] = a.par[i] + b.par[i];
    }
    return c;
}

__host__ __device__ Walker ScaleWalker ( Walker a, float s )
{
    Walker c;
    for ( int i = 0; i < NPRS; i++ )
    {
        c.par[i] = s * a.par[i];
    }
    return c;
}

__host__ __device__ Complex AddComplex ( Complex a, Complex b )
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__host__ __device__ Complex ScaleComplex ( Complex a, float s )
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

__host__ __device__ Complex MultiplyComplex ( Complex a, Complex b )
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__host__ __device__ Complex ConjugateComplex ( Complex a )
{
    Complex c;
    c.x = a.x;
    c.y = - a.y;
    return c;
}

__host__ __device__ float PowerLaw ( const float phtnIndx, const float nrmlztn, const float enrgLwr, const float enrgHghr )
{
    float flx;
    if ( fabsf ( 1 - phtnIndx ) > TLR )
    {
        flx = powf ( 10, nrmlztn ) * ( powf ( enrgHghr, 1 - phtnIndx ) - powf ( enrgLwr, 1 - phtnIndx ) ) / ( 1 - phtnIndx );
    }
    else
    {
        flx = powf ( 10, nrmlztn ) * ( logf ( enrgHghr ) - logf ( enrgLwr ) );
    }
    return flx;
}

__host__ __device__ float BlackBody ( const float kT, const float lgRtD, const float enrgLwr, const float enrgHghr )
{
    float t, anorm, elow, x, tinv, anormh, alow, ehi, ahi, flx;

    t = kT;
    tinv = 1. / t;
    anorm = 1.0344e-3f * 1e8f * powf ( 10, 2 * lgRtD ) ;
    anormh = 0.5 * anorm;
    elow = enrgLwr;
    x = elow * tinv;
    if ( x <= 1.0e-4f )
    {
        alow = elow * t;
    }
    else if ( x > 60.0 )
    {
        flx = 0;
        return flx;
    }
    else
    {
        alow = elow * elow / ( expf ( x ) - 1.0e0f );
    }
    ehi = enrgHghr;
    x = ehi * tinv;
    if ( x <= 1.0e-4f )
    {
        ahi = ehi * t;
    }
    else if ( x > 60.0 )
    {
        flx = 0;
        return flx;
    }
    else
    {
        ahi = ehi * ehi / ( expf ( x ) - 1.0e0f );
    }
    flx = anormh * ( alow + ahi ) * ( ehi - elow );
    return flx;
}

__host__ __device__ float Poisson ( const float srcCnts, const float flddMdlFlx, const float srcExptm )
{
    float sttstc;
    float mdlCnts = srcExptm * flddMdlFlx;
    if ( ( srcCnts != 0 ) && ( fabsf ( mdlCnts ) > TLR ) )
    {
        sttstc = 2. * ( mdlCnts - srcCnts + srcCnts * ( logf ( srcCnts ) - logf ( mdlCnts ) ) );
    }
    else if ( ( srcCnts == 0 ) && ( fabsf ( mdlCnts ) > TLR ) )
    {
        sttstc = 2. * mdlCnts;
    }
    else
    {
        sttstc = 0;
    }
    return sttstc;
}

__host__ __device__ int FindElementIndex ( const float *xx, const int n, const float x )
{
    int ju, jm, jl, jres;
    jl = 0;
    ju = n;
    while ( ju - jl > 1 )
    {
        jm = floorf ( 0.5 * ( ju + jl ) );
        if ( x >= xx[jm] ) { jl = jm; } else { ju = jm; }
    }
    jres = jl;
    if ( x == xx[0] ) jres = 0;
    if ( x >= xx[n-1] ) jres = n - 1;
    return jres;
}

__host__ void AssembleArrayOfPhotoelectricCrossections ( const int nmbrOfEnrgChnnls, const int nmbrOfElmnts, int sgFlag,
                                                         float *enrgChnnls, int *atmcNmbrs,
                                                         float *crssctns )
{
    int status = 0, versn = sgFlag, indx;
    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        for ( int j = 0; j < nmbrOfElmnts; j++ )
        {
            indx = j + i * nmbrOfElmnts;
            crssctns[indx] = photo_ ( &enrgChnnls[i], &enrgChnnls[i+1], &atmcNmbrs[j], &versn, &status );
        }
    }
}

__host__ void ReadLastPositionOfWalkersFromFile ( const char *thrdNm, const int indx, const int nmbrOfWlkrs,
                                                  float *lstChn )
{
    FILE *flPntr;
    char flNm[FLEN_CARD];
    float value;
    int i = 0, k = 0, j;
    snprintf ( flNm, sizeof ( flNm ), "%s%i", thrdNm, indx );
    flPntr = fopen ( flNm, "r" );
    while ( fscanf ( flPntr, "%e", &value ) == 1 )
    {
        i += 1;
    }
    fclose ( flPntr );
    flPntr = fopen ( flNm, "r" );
    while ( fscanf ( flPntr, "%e", &value ) == 1 )
    {
        if ( k >= i - nmbrOfWlkrs * ( NPRS + 1 ) )
        {
            j = k - ( i - nmbrOfWlkrs * ( NPRS + 1 ) );
            lstChn[j] = value;
        }
        k += 1;
    }
    fclose ( flPntr );
}

__host__ void WriteChainToFile ( const char *thrdNm, const int indx,
                                 const int nmbrOfWlkrs, const int nmbrOfStps, const Walker *chnOfWlkrs, const float *chnOfSttstcs )
{
    FILE *flPntr;
    char flNm[FLEN_CARD];
    int ttlChnIndx, stpIndx, wlkrIndx, prmtrIndx;
    snprintf ( flNm, sizeof ( flNm ), "%s%i", thrdNm, indx );
    flPntr = fopen ( flNm, "w" );
    stpIndx = 0;
    while ( stpIndx < nmbrOfStps )
    {
        wlkrIndx = 0;
        while ( wlkrIndx < nmbrOfWlkrs )
        {
            ttlChnIndx = wlkrIndx + stpIndx * nmbrOfWlkrs;
            prmtrIndx = 0;
            while ( prmtrIndx < NPRS )
            {
                fprintf ( flPntr, " %.8E ", chnOfWlkrs[ttlChnIndx].par[prmtrIndx] );
                prmtrIndx += 1;
            }
            fprintf ( flPntr, " %.8E\n", chnOfSttstcs[ttlChnIndx] );
            wlkrIndx += 1;
        }
        stpIndx += 1;
    }
    fclose ( flPntr );
}

__host__ void SimpleReadNsaTable ( const char *flNm, const int numEn, const int numTe, float *data, float *Te, float *En, float *fluxes )
{
    FILE *flPntr;
    float value;
    int i = 0;
    flPntr = fopen ( flNm, "r" );
    while ( fscanf ( flPntr, "%e", &value ) == 1 )
    {
        data[i] = value;
        i += 1;
    }
    for (int j = 0; j < numEn; j++)
    {
        En[j] = data[(j+1)*(numTe+1)];
    }
    for (int j = 0; j < numTe; j++)
    {
        Te[j] = data[j+1];
    }
    for (int j = 0; j < numEn; j++)
    {
         for (int i = 0; i < numTe; i++)
         {
             fluxes[j+i*numEn] = data[(i+1)+(j+1)*(numTe+1)];
         }
    }
    fclose ( flPntr );
}

__host__ void SimpleReadReddenningData ( const char *flNm, const int numDist, float *data, float *Dist, float *EBV, float *errDist, float *errEBV )
{
    FILE *flPntr;
    float value;
    int i = 0;
    flPntr = fopen ( flNm, "r" );
    while ( fscanf (flPntr, "%e", &value ) == 1 )
    {
        data[i] = value;
        i += 1;
    }
    for ( int j = 0; j < numDist; j++ )
    {
        Dist[j] = data[4*j];
        EBV[j] = data[4*j+1];
        errDist[j] = data[4*j+2];
        errEBV[j] = data[4*j+3];
    }
    fclose ( flPntr );
}

__host__ void SimpleReadDataFloat ( const char *flNm, float *data )
{
    FILE *flPntr;
    float value;
    int i = 0;
    flPntr = fopen ( flNm, "r" );
    while ( fscanf ( flPntr, "%e", &value ) == 1 )
    {
        data[i] = value;
        i += 1;
    }
    fclose ( flPntr );
}

__host__ void SimpleReadDataInt ( const char *flNm, int *data )
{
    FILE *flPntr;
    int value;
    int i = 0;
    flPntr = fopen ( flNm, "r" );
    while ( fscanf ( flPntr, "%i", &value ) == 1 )
    {
        data[i] = value;
        i += 1;
    }
    fclose ( flPntr );
}

__host__ void SimpleWriteDataFloat ( const char *flNm, const int nmbrOfStps, const float *chn )
{
    FILE *flPntr;
    flPntr = fopen ( flNm, "w" );
    for ( int i = 0; i < nmbrOfStps; i++ )
    {
        fprintf ( flPntr, " %.8E\n", chn[i] );
    }
    fclose ( flPntr );
}


__host__ void SimpleWriteDataFloat2D ( const char *flNm, const int nmbrOfStps, const int nmbrOfWlkrs, const float *chn )
{
    FILE *flPntr;
    flPntr = fopen ( flNm, "w" );
    for ( int j = 0; j < nmbrOfStps; j++ )
    {
        for ( int i = 0; i < nmbrOfWlkrs; i++ )
        {
            fprintf ( flPntr, " %.8E ", chn[i+j*nmbrOfWlkrs] );
        }
        fprintf ( flPntr,  "\n" );
    }
    fclose ( flPntr );
}

__host__ void AutocorrelationFunctionAveraged ( cufftResult_t cufftRes, cublasStatus_t cublasStat, cublasHandle_t cublasHandle, cufftHandle cufftPlan,
                                                const int nmbrOfStps, const int nmbrOfWlkrs, const float *chnFnctn, float *atcrrFnctn )
{
    int incxx = INCXX, incyy = INCYY;
    float alpha = ALPHA, beta = BETA;
    int thrdsPerBlck = THRDSPERBLCK;
    dim3 dimBlock ( thrdsPerBlck, thrdsPerBlck );
    int blcksPerThrd_0 = ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck;
    int blcksPerThrd_1 = ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck;
    dim3 dimGrid_0 ( ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_1 ( ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfWlkrs + thrdsPerBlck - 1 ) / thrdsPerBlck );
    dim3 dimGrid_2 ( ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck, ( nmbrOfStps + thrdsPerBlck - 1 ) / thrdsPerBlck );

    float *stps, *smOfChn, *cntrlChnFnctn, *wlkrs, *cmSmMtrx;
    cufftComplex *ftOfChn;
    cudaMallocManaged ( ( void ** ) &stps, nmbrOfStps * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &smOfChn, nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &cntrlChnFnctn, nmbrOfStps * nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &wlkrs, nmbrOfWlkrs * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &ftOfChn, nmbrOfStps * nmbrOfWlkrs * sizeof ( cufftComplex ) );
    cudaMallocManaged ( ( void ** ) &cmSmMtrx, nmbrOfStps * nmbrOfStps * sizeof ( float ) );

    ReturnConstantArray <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfStps, alpha / nmbrOfStps, stps );
    cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_N, nmbrOfWlkrs, nmbrOfStps, &alpha, chnFnctn, nmbrOfWlkrs, stps, incxx, &beta, smOfChn, incyy );
    if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: " ); }
    ReturnCentralChainFunction <<< dimGrid_0, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, smOfChn, chnFnctn, cntrlChnFnctn );

    ReturnChainFunctionTest <<< dimGrid_0, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 0, cntrlChnFnctn, ftOfChn );
    cufftRes = cufftExecC2C ( cufftPlan, ( cufftComplex * ) ftOfChn, ( cufftComplex * ) ftOfChn, CUFFT_FORWARD );
    if ( cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error:" ); }
    ComplexPointwiseMultiplyByConjugateAndScale <<< dimGrid_1, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, alpha / nmbrOfStps, ftOfChn );
    cufftRes = cufftExecC2C ( cufftPlan, ( cufftComplex * ) ftOfChn, ( cufftComplex * ) ftOfChn, CUFFT_INVERSE );
    if ( cufftRes != CUFFT_SUCCESS ) { fprintf ( stderr, "CUFFT error: " ); }
    ReturnChainFunctionTest <<< dimGrid_0, dimBlock >>> ( nmbrOfStps, nmbrOfWlkrs, 1, cntrlChnFnctn, ftOfChn );

    ReturnConstantArray <<< blcksPerThrd_1, thrdsPerBlck >>> ( nmbrOfWlkrs, alpha / nmbrOfWlkrs, wlkrs );
    cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, nmbrOfWlkrs, nmbrOfStps, &alpha, cntrlChnFnctn, nmbrOfWlkrs, wlkrs, incxx, &beta, atcrrFnctn, incyy );
    if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: " ); }

    NormalizeChain <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfStps, atcrrFnctn );
    //MakeMatrix <<< dimGrid_2, dimBlock >>> ( nmbrOfStps, atcrrFnctn, cmSmMtrx );
    //ReturnConstantArray <<< blcksPerThrd_0, thrdsPerBlck >>> ( nmbrOfStps, alpha, stps );
    //cublasStat = cublasSgemv ( cublasHandle, CUBLAS_OP_T, nmbrOfStps, nmbrOfStps, &alpha, cmSmMtrx, nmbrOfStps, stps, incxx, &beta, cmSmAtcrrFnctn, incyy );
    //if ( cublasStat != CUBLAS_STATUS_SUCCESS ) { fprintf ( stderr, " CUBLAS error: " ); }

    cudaFree ( stps );
    cudaFree ( smOfChn );
    cudaFree ( cntrlChnFnctn );
    cudaFree ( wlkrs );
    cudaFree ( ftOfChn );
    cudaFree ( cmSmMtrx );
}

__host__ void CumulativeSumOfAutocorrelationFunction ( const int nmbrOfStps, const float *chn, float *cmSmChn )
{
    float sum = 0;
    for ( int i = 0; i < nmbrOfStps; i++ )
    {
        sum = sum + chn[i];
        //printf ( " %.8E", sum );
        cmSmChn[i] = sum;
        //printf ( " %.8E\n", cmSmChn[i] );
    }
}

__host__ int ChooseWindow ( const int nmbrOfStps, const float c, const float *cmSmChn )
{
    int m = 0;
    while ( ( m < c * ( 2 * cmSmChn[m] - 1e0f ) ) && ( m < nmbrOfStps )  )
    {
        m += 1;
    }
    return m;
}

/* Kernels: */
__global__ void InitializeWalkersAtRandom ( const int nmbrOfWlkrs, const float dlt, Walker strtngWlkr, const float *rndmVls,
                                            Walker *wlkrs )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( wlIndx < nmbrOfWlkrs )
    {
        wlkrs[wlIndx] = AddWalkers ( strtngWlkr, ScaleWalker ( strtngWlkr, dlt * rndmVls[wlIndx] ) );
    }
}

__global__ void InitializeWalkersAndStatisticsFromLastChain ( const int nmbrOfWlkrs, const float *lstChn,
                                                              Walker *wlkrs, float *sttstcs )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int k = ( NPRS + 1 ) * wlIndx;
    int prIndx, chIndx;
    if ( wlIndx < nmbrOfWlkrs )
    {
        prIndx = 0;
        while ( prIndx < NPRS )
        {
            chIndx = prIndx + k;
            wlkrs[wlIndx].par[prIndx] = lstChn[chIndx];
            prIndx += 1;
        }
        chIndx = prIndx + k;
        sttstcs[wlIndx] = lstChn[chIndx];
    }
}

__global__ void WriteWalkersAndStatisticsToChain ( const int nmbrOfWlkrs, const int stpIndx,
                                                   const Walker *wlkrs, const float *sttstcs,
                                                   Walker *chnOfWlkrs, float *chnOfSttstcs )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int ttIndx = wlIndx + stpIndx * nmbrOfWlkrs;
    if ( wlIndx < nmbrOfWlkrs )
    {
        chnOfWlkrs[ttIndx] = wlkrs[wlIndx];
        chnOfSttstcs[ttIndx] = sttstcs[wlIndx];
    }
}

__global__ void AssembleArrayOfPriors ( const int nmbrOfWlkrs, const Walker *wlkrs, const float *mNh, const float *sNh, float *prrs )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( wlIndx < nmbrOfWlkrs )
    {
        prrs[wlIndx] = PriorStatistic ( wlkrs[wlIndx], PriorCondition ( wlkrs[wlIndx] ), mNh[wlIndx], sNh[wlIndx] );
    }
}

__global__ void AssembleArrayOfNoticedChannels ( const int nmbrOfChnnls, const float lwrNtcdEnrg, const float hghrNtcdEnrg,
                                                 const float *lwrChnnlBndrs, const float *hghrChnnlBndrs, const float *gdQltChnnls,
                                                 float *ntcdChnnls )
{
    int chIndx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( chIndx < nmbrOfChnnls )
    {
        ntcdChnnls[chIndx] = ( lwrChnnlBndrs[chIndx] > lwrNtcdEnrg ) * ( hghrChnnlBndrs[chIndx] < hghrNtcdEnrg ) * ( 1 - gdQltChnnls[chIndx] );
    }
}

__global__ void AssembleArrayOfChannelStatistics ( const int nmbrOfWlkrs, const int nmbrOfChnnls, const float srcExptm, const float bckgrndExptm,
                                                   const float *srcCnts, const float *bckgrndCnts, const float *flddMdlFlxs,
                                                   float *chnnlSttstcs )
{
    int chIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
    int ttIndx = chIndx + wlIndx * nmbrOfChnnls;
    if ( ( chIndx < nmbrOfChnnls ) && ( wlIndx < nmbrOfWlkrs ) )
    {
        chnnlSttstcs[ttIndx] = Poisson ( srcCnts[chIndx], flddMdlFlxs[ttIndx], srcExptm );
    }
}

__global__ void GenerateProposal ( const int nmbrOfHlfTheWlkrs, const int stpIndx, const int sbstIndx,
                                   const Walker *wlkrs, const float *rndmVls,
                                   float *zRndmVls, Walker *prpsdWlkrs )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int ttSbIndx = wlIndx + sbstIndx * nmbrOfHlfTheWlkrs;
    int rnIndx, ttRnIndx, ttCmSbIndx, k;
    float zz;
    Walker B;
    if ( wlIndx < nmbrOfHlfTheWlkrs )
    {
        rnIndx = 0;
        ttRnIndx = wlIndx + rnIndx * nmbrOfHlfTheWlkrs + stpIndx * 3 * nmbrOfHlfTheWlkrs;
        zz = 0.5 * powf ( rndmVls[ttRnIndx] + 1, 2. );
        zRndmVls[wlIndx] = zz;
        rnIndx = 1;
        ttRnIndx = wlIndx + rnIndx * nmbrOfHlfTheWlkrs + stpIndx * 3 * nmbrOfHlfTheWlkrs;
        k = ( int ) truncf ( rndmVls[ttRnIndx] * ( nmbrOfHlfTheWlkrs - 1 + 0.999999 ) );
        ttCmSbIndx = k + ( 1 - sbstIndx ) * nmbrOfHlfTheWlkrs;
        B = AddWalkers ( wlkrs[ttSbIndx], ScaleWalker ( wlkrs[ttCmSbIndx], -1. ) );
        prpsdWlkrs[wlIndx] = AddWalkers ( wlkrs[ttCmSbIndx], ScaleWalker ( B, zz ) );
    }
}

__global__ void UpdateWalkers ( const int nmbrOfHlfTheWlkrs, const int stpIndx, const int sbstIndx,
                                const Walker *prpsdWlkrs, const float *prpsdSttstcs, const float *prrs, const float *zRndmVls, const float *rndmVls,
                                Walker *wlkrs, float *sttstcs )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int ttSbIndx = wlIndx + sbstIndx * nmbrOfHlfTheWlkrs;
    int rnIndx = 2;
    int ttRnIndx = wlIndx + rnIndx * nmbrOfHlfTheWlkrs + stpIndx * 3 * nmbrOfHlfTheWlkrs;
    float q;
    if ( wlIndx < nmbrOfHlfTheWlkrs )
    {
        q = - 0.5 * ( prpsdSttstcs[wlIndx] + prrs[wlIndx] - sttstcs[ttSbIndx] );
        q = expf ( q ) * powf ( zRndmVls[wlIndx], NPRS - 1 );
        if ( q > rndmVls[ttRnIndx] )
        {
            wlkrs[ttSbIndx] = prpsdWlkrs[wlIndx];
            sttstcs[ttSbIndx] = prpsdSttstcs[wlIndx];
        }
    }
}

__global__ void ComplexPointwiseMultiplyByConjugateAndScale ( const int nmbrOfStps, const int nmbrOfWlkrs, const float scl, Complex *a )
{
    int stIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
    int ttIndx = stIndx + wlIndx * nmbrOfStps;
    if ( ( wlIndx < nmbrOfWlkrs ) && ( stIndx < nmbrOfStps ) )
    {
        a[ttIndx] = ScaleComplex ( MultiplyComplex ( a[ttIndx], ConjugateComplex ( a[ttIndx] ) ), scl );
    }
}

__global__ void ReturnChainFunctionTest ( const int nmbrOfStps, const int nmbrOfWlkrs, const int sw, float *chn, Complex *a  )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int stIndx = threadIdx.y + blockDim.y * blockIdx.y;
    int ttIndx0 = wlIndx + stIndx * nmbrOfWlkrs;
    int ttIndx1 = stIndx + wlIndx * nmbrOfStps;
    if ( ( wlIndx < nmbrOfWlkrs ) && ( stIndx < nmbrOfStps ) )
    {
        if ( sw == 0 ) { a[ttIndx1].x = chn[ttIndx0]; a[ttIndx1].y = 0; }
        else if ( sw == 1 ) { chn[ttIndx0] = a[ttIndx1].x; }

    }
}

__global__ void ReturnChainFunction ( const int nmbrOfStps, const int nmbrOfWlkrs, const int prmtrIndx, const Walker *chnOfWlkrs, float *chnFnctn )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int stIndx = threadIdx.y + blockDim.y * blockIdx.y;
    int ttIndx = wlIndx + stIndx * nmbrOfWlkrs;
    if ( ( wlIndx < nmbrOfWlkrs ) && ( stIndx < nmbrOfStps ) )
    {
        chnFnctn[ttIndx] = chnOfWlkrs[ttIndx].par[prmtrIndx];
    }
}

__global__ void ReturnConstantArray ( const int nmbrOfStps, const float cnst, float *cnstArr )
{
    int stIndx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( stIndx < nmbrOfStps )
    {
        cnstArr[stIndx] = cnst;
    }
}

__global__ void ReturnCentralChainFunction ( const int nmbrOfStps, const int nmbrOfWlkrs, const float *smOfChnFnctn, const float *chnFnctn, float *cntrlChnFnctn )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int stIndx = threadIdx.y + blockDim.y * blockIdx.y;
    int ttIndx = wlIndx + stIndx * nmbrOfWlkrs;
    if ( ( wlIndx < nmbrOfWlkrs ) && ( stIndx < nmbrOfStps )  )
    {
        cntrlChnFnctn[ttIndx] = chnFnctn[ttIndx] - smOfChnFnctn[wlIndx];
    }
}

__global__ void NormalizeChain ( const int nmbrOfStps, float *chn )
{
    int stIndx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( stIndx < nmbrOfStps )
    {
        chn[stIndx] = chn[stIndx] / chn[0];
    }
}

__global__ void MakeMatrix ( const int nmbrOfStps, const float *chn, float *cmSmMtrx )
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.x;
    if ( ( i < nmbrOfStps ) && ( j < nmbrOfStps ) )
    {
        cmSmMtrx[i+j*nmbrOfStps] = ( i <= j ) * chn[i];
    }
}

__global__ void BilinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int prmtrIndx,
                                        const float *data, const float *xin, const float *yin, const int M1, const int M2,
                                        const float *enrgChnnls, const Walker *wlkrs,
                                        float *mdlFlxs )
{
    int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
    int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
    float xxout, yyout, sa, gr, NormD, DimConst, a, b, d00, d01, d10, d11, tmp1, tmp2, tmp3;
    int v, w;
    if ( ( enIndx < nmbrOfEnrgChnnls ) && ( wlIndx < nmbrOfWlkrs ) )
    {
        gr = sqrtf ( 1.0 - 2.952 * MNS / RNS );
        xxout = 0.5 * ( enrgChnnls[enIndx] + enrgChnnls[enIndx+1] ) / gr;
        yyout = wlkrs[wlIndx].par[prmtrIndx];
        sa = powf ( RNS / gr, 2. );
        NormD = - 2 * ( wlkrs[wlIndx].par[prmtrIndx+1] );
        DimConst = 2 * KMCMPCCM;
        v = FindElementIndex ( xin, M1, xxout );
        w = FindElementIndex ( yin, M2, yyout );
        a = ( xxout - xin[v] ) / ( xin[v+1] - xin[v] );
        b = ( yyout - yin[w] ) / ( yin[w+1] - yin[w] );
        //float INFi = 1e10f;
        if ( ( v < M1 ) && ( w < M2 ) ) d00 = data[w*M1+v]; else d00 = 0; //-INFi;
        if ( ( v+1 < M1 ) && ( w < M2 ) ) d10 = data[w*M1+v+1]; else d10 = 0; // -INFi;
        if ( ( v < M1 ) && ( w+1 < M2 ) ) d01 = data[(w+1)*M1+v]; else d01 = 0; // -INFi;
        if ( ( v+1 < M1 ) && ( w+1 < M2 ) ) d11 = data[(w+1)*M1+v+1]; else d11 = 0; // -INFi;
        tmp1 = a * d10 + ( -d00 * a + d00 );
        tmp2 = a * d11 + ( -d01 * a + d01 );
        tmp3 = b * tmp2 + ( -tmp1 * b + tmp1 );
        mdlFlxs[enIndx+wlIndx*nmbrOfEnrgChnnls] = tmp3 * sa * powf ( 10., NormD + DimConst ) * ( enrgChnnls[enIndx+1] - enrgChnnls[enIndx] );
    }
}

__global__ void LinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int prmtrIndx,
                                      const float *Dist, const float *EBV, const float *errEBV, const Walker *wlkrs,
                                      float *mNh, float *sNh )
{
    int wlIndx = threadIdx.x + blockDim.x * blockIdx.x;
    float xxout, a, dmNh0, dmNh1, dsNh0, dsNh1, tmpMNh, tmpSNh;
    int v;
    if ( wlIndx < nmbrOfWlkrs )
    {
        xxout = powf ( 10, wlkrs[wlIndx].par[prmtrIndx] );
        v = FindElementIndex ( Dist, nmbrOfDistBins, xxout );
        a = ( xxout - Dist[v] ) / ( Dist[v+1] - Dist[v] );
        dmNh0 = EBV[v];
        dmNh1 = EBV[v+1];
        dsNh0 = errEBV[v];
        dsNh1 = errEBV[v+1];
        tmpMNh = a * dmNh1 + ( -dmNh0 * a + dmNh0 );
        tmpSNh = a * dsNh1 + ( -dsNh0 * a + dsNh0 );
        mNh[wlIndx] = 0.8 * tmpMNh;
        sNh[wlIndx] = 0.8 * tmpMNh * ( tmpSNh / tmpMNh + 0.3 / 0.8 );
    }
}

__host__ void FreeSpec ( const Spectrum *spec )
{
  for ( int i = 0; i < NSPCTR; i++ )
  {
    cudaFree ( spec[i].rmfVlsInCsc );
    cudaFree ( spec[i].rmfIndxInCsc );
    cudaFree ( spec[i].rmfPntrInCsc );
    cudaFree ( spec[i].rmfVls );
    cudaFree ( spec[i].rmfIndx );
    cudaFree ( spec[i].rmfPntr );
    cudaFree ( spec[i].enrgChnnls );
    cudaFree ( spec[i].arfFctrs );
    cudaFree ( spec[i].srcCnts );
    cudaFree ( spec[i].bckgrndCnts );
    cudaFree ( spec[i].gdQltChnnls );
    cudaFree ( spec[i].lwrChnnlBndrs );
    cudaFree ( spec[i].hghrChnnlBndrs );
    cudaFree ( spec[i].crssctns );
    cudaFree ( spec[i].absrptnFctrs );
    cudaFree ( spec[i].mdlFlxs );
    cudaFree ( spec[i].flddMdlFlxs );
    cudaFree ( spec[i].chnnlSttstcs );
    cudaFree ( spec[i].ntcdChnnls );

  }
}

__host__ void ReadAllTheFitsData ( const char *spcLst[NSPCTR], const int verbose, Spectrum *spec )
{
  char srcTbl[FLEN_CARD], arfTbl[FLEN_CARD], rmfTbl[FLEN_CARD], bckgrndTbl[FLEN_CARD];
  for ( int i = 0; i < NSPCTR; i++ )
  {
    ReadFitsInfo ( spcLst[i], &spec[i].nmbrOfEnrgChnnls, &spec[i].nmbrOfChnnls, &spec[i].nmbrOfRmfVls, &spec[i].srcExptm, &spec[i].bckgrndExptm, srcTbl, arfTbl, rmfTbl, bckgrndTbl );
    if ( verbose == 1 )
    {
      printf ( ".................................................................\n" );
      printf ( " Spectrum number  -- %i\n", i );
      printf ( " Spectrum table   -- %s\n", srcTbl );
      printf ( " ARF table        -- %s\n", arfTbl );
      printf ( " RMF table        -- %s\n", rmfTbl );
      printf ( " Background table -- %s\n", bckgrndTbl );
      printf ( " Number of energy channels                = %i\n", spec[i].nmbrOfEnrgChnnls );
      printf ( " Number of instrument channels            = %i\n", spec[i].nmbrOfChnnls );
      printf ( " Number of nonzero elements of RMF matrix = %i\n", spec[i].nmbrOfRmfVls );
      printf ( " Exposure time                            = %.8E\n", spec[i].srcExptm );
      printf ( " Exposure time (background)               = %.8E\n", spec[i].bckgrndExptm );
    }
    cudaMallocManaged ( ( void ** ) &spec[i].rmfPntrInCsc, ( spec[i].nmbrOfEnrgChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec[i].rmfIndxInCsc, spec[i].nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec[i].rmfPntr, ( spec[i].nmbrOfChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec[i].rmfIndx, spec[i].nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec[i].rmfVlsInCsc, spec[i].nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].rmfVls, spec[i].nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].enrgChnnls, ( spec[i].nmbrOfEnrgChnnls + 1 ) * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].arfFctrs, spec[i].nmbrOfEnrgChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].srcCnts, spec[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].bckgrndCnts, spec[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].lwrChnnlBndrs, spec[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].hghrChnnlBndrs, spec[i].nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec[i].gdQltChnnls, spec[i].nmbrOfChnnls * sizeof ( float ) );
    ReadFitsData ( srcTbl, arfTbl, rmfTbl, bckgrndTbl, spec[i].nmbrOfEnrgChnnls, spec[i].nmbrOfChnnls, spec[i].nmbrOfRmfVls, spec[i].srcCnts, spec[i].bckgrndCnts, spec[i].arfFctrs, spec[i].rmfVlsInCsc, spec[i].rmfIndxInCsc, spec[i].rmfPntrInCsc, spec[i].gdQltChnnls, spec[i].lwrChnnlBndrs, spec[i].hghrChnnlBndrs, spec[i].enrgChnnls );
  }
}

__host__ void ReadFitsInfo ( const char *spcFl,
                    int *nmbrOfEnrgChnnls, int *nmbrOfChnnls, int *nmbrOfRmfVls, float *srcExptm, float *bckgrndExptm,
                    char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] )
{
    fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
    int status = 0, intnull = 0, anynull = 0, colnum;
    char card[FLEN_CARD], colNgr[] = "N_GRP", colNch[] = "N_CHAN";
    float floatnull;

    /* Open Spectrum  */
    snprintf ( srcTbl, sizeof ( card ), "%s%s", spcFl, "[SPECTRUM]" );
    fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TINT, "NAXIS2", nmbrOfChnnls, NULL, &status );
    fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", srcExptm, NULL, &status );
    fits_read_key ( ftsPntr, TSTRING, "ANCRFILE", card, NULL, &status );
    snprintf ( arfTbl, sizeof ( card ), "%s%s", card, "[SPECRESP]" );
    fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
    snprintf ( rmfTbl, sizeof ( card ), "%s%s", card, "[MATRIX]" );
    fits_read_key ( ftsPntr, TSTRING, "BACKFILE", card, NULL, &status );
    snprintf ( bckgrndTbl, sizeof ( card ), "%s%s", card, "[SPECTRUM]" );
    fits_read_key ( ftsPntr, TSTRING, "BACKFILE", card, NULL, &status );

    /* Open Background file */
    fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", bckgrndExptm, NULL, &status );

    /* Open RMF file */
    fits_open_file ( &ftsPntr, rmfTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TINT, "NAXIS2", nmbrOfEnrgChnnls, NULL, &status );

    int *n_grp;
    n_grp = ( int * ) malloc ( *nmbrOfEnrgChnnls * sizeof ( int ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colNgr, &colnum, &status );
    fits_read_col_int ( ftsPntr, colnum, 1, 1, *nmbrOfEnrgChnnls, intnull, n_grp, &anynull, &status );

    int *n_chan_vec;
    n_chan_vec = ( int * ) malloc ( *nmbrOfChnnls * sizeof ( int ) );

    int sum = 0;

    for ( int i = 0; i < *nmbrOfEnrgChnnls; i++ )
    {
        fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
        fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, n_chan_vec, &anynull, &status );

        for ( int j = 0; j < n_grp[i]; j++ )
        {
            sum = sum + n_chan_vec[j];
        }
    }

    *nmbrOfRmfVls = sum;

    free ( n_chan_vec );
    free ( n_grp );
}

__host__ void ReadFitsData ( const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD],
                    const int nmbrOfEnrgChnnls, const int nmbrOfChnnls, const int nmbrOfRmfVls,
                    float *srcCnts, float *bckgrndCnts, float *arfFctrs, float *rmfVlsInCsc, int *rmfIndxInCsc, int *rmfPntrInCsc,
                    float *gdQltChnnls, float *lwrChnnlBndrs, float *hghrChnnlBndrs, float *enrgChnnls )
{
    fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
    int status = 0, anynull, colnum, intnull = 0, rep_chan = 6;
    char card[FLEN_CARD], EboundsTable[FLEN_CARD], Telescop[FLEN_CARD];
    char colNgr[]="N_GRP",colNch[]="N_CHAN",colFch[]="F_CHAN",colCounts[]="COUNTS",colSpecResp[]="SPECRESP",colEnLo[]="ENERG_LO",colEnHi[]="ENERG_HI",colMat[]="MATRIX",colEmin[]="E_MIN",colEmax[]="E_MAX";
    float floatnull, backscal_src, backscal_bkg;

    /* Read Spectrum: */
    fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
    snprintf ( EboundsTable, sizeof ( EboundsTable ), "%s%s", card, "[EBOUNDS]" );
    fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", &backscal_src, NULL, &status );
    fits_read_key ( ftsPntr, TSTRING, "TELESCOP", Telescop, NULL, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, srcCnts, &anynull, &status );

    /* Read ARF FILE: */
    fits_open_file ( &ftsPntr, arfTbl, READONLY, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colSpecResp, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, arfFctrs, &anynull, &status );

    /* Read Background: */
    fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", &backscal_bkg, NULL, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, bckgrndCnts, &anynull, &status );

    for ( int i = 0; i < nmbrOfChnnls; i++ )
    {
        bckgrndCnts[i] = bckgrndCnts[i] * backscal_src / backscal_bkg;;
    }

    /* Read RMF file */
    fits_open_file ( &ftsPntr, rmfTbl, READONLY, &status );

    float *enelo_vec, *enehi_vec;
    enelo_vec = ( float * ) malloc ( nmbrOfEnrgChnnls * sizeof ( float ) );
    enehi_vec = ( float * ) malloc ( nmbrOfEnrgChnnls * sizeof ( float ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colEnLo, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, enelo_vec, &anynull, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colEnHi, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, enehi_vec, &anynull, &status );

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        enrgChnnls[i] = enelo_vec[i];
    }

    enrgChnnls[nmbrOfEnrgChnnls] = enehi_vec[nmbrOfEnrgChnnls-1];

    int *f_chan_vec, *n_chan_vec;
    f_chan_vec = ( int * ) malloc ( rep_chan * sizeof ( int ) );
    n_chan_vec = ( int * ) malloc ( rep_chan * sizeof ( int ) );

    int *f_chan, *n_chan;
    f_chan = ( int * ) malloc ( rep_chan * nmbrOfEnrgChnnls * sizeof ( int ) );
    n_chan = ( int * ) malloc ( rep_chan * nmbrOfEnrgChnnls * sizeof ( int ) );

    int *n_grp;
    n_grp = ( int * ) malloc ( nmbrOfEnrgChnnls * sizeof ( int ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colNgr, &colnum, &status );
    fits_read_col_int ( ftsPntr, colnum, 1, 1, nmbrOfEnrgChnnls, intnull, n_grp, &anynull, &status );

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
        fits_read_col_int ( ftsPntr, colnum, i+1, 1, n_grp[i], intnull, n_chan_vec, &anynull, &status );

        for ( int j = 0; j < rep_chan; j++ )
        {
            n_chan[i*rep_chan+j] = n_chan_vec[j];
        }
    }

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        fits_get_colnum ( ftsPntr, CASEINSEN, colFch, &colnum, &status );
        fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, f_chan_vec, &anynull, &status );

        for ( int j = 0; j < rep_chan; j++ )
        {
            f_chan[i*rep_chan+j] = f_chan_vec[j];
        }
    }

    int sum = 0;
    rmfPntrInCsc[0] = 0;

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        for ( int j = 0; j < n_grp[i]; j++ )
        {
            sum = sum + n_chan[rep_chan*i+j];
        }
        rmfPntrInCsc[i+1] = sum;
    }

    int m = 0;

    if ( nmbrOfChnnls == 4096 )
    {
        for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
        {
            for ( int j = 0; j < n_grp[i]; j++ )
            {
                for ( int k = f_chan[rep_chan*i+j] ; k < f_chan[rep_chan*i+j] + n_chan[rep_chan*i+j]; k++ )
                {
                    rmfIndxInCsc[m] = k;
                    m = m + 1;
                }
            }
        }
    }
    else if ( nmbrOfChnnls == 1024 )
    {
        for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
        {
            for ( int j = 0; j < n_grp[i]; j++ )
            {
                for ( int k = f_chan[rep_chan*i+j] - 1; k < f_chan[rep_chan*i+j] - 1 + n_chan[rep_chan*i+j]; k++ )
                {
                    rmfIndxInCsc[m] = k;
                    m = m + 1;
                }
            }
        }
    }

    float *rmf_vec;
    rmf_vec = ( float * ) malloc ( nmbrOfChnnls * sizeof ( float ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colMat, &colnum, &status );

    m = 0;

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        sum = rmfPntrInCsc[i+1] - rmfPntrInCsc[i];

        fits_read_col ( ftsPntr, TFLOAT, colnum, i+1, 1, sum, &floatnull, rmf_vec, &anynull, &status );

        for ( int k = 0; k < sum; k++ )
        {
            rmfVlsInCsc[m] = rmf_vec[k];
            //printf ( "%.8E\n", rmfVlsInCsc[m] );
            m = m + 1;
        }
    }

    /* Read Ebounds Table: */
    fits_open_file ( &ftsPntr, EboundsTable, READONLY, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colEmin, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, lwrChnnlBndrs, &anynull, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colEmax, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, hghrChnnlBndrs, &anynull, &status );

    free ( enelo_vec );
    free ( enehi_vec );
    free ( rmf_vec );
    free ( f_chan_vec );
    free ( n_chan_vec );
    free ( n_chan );
    free ( f_chan );
    free ( n_grp );
}

#endif // _GWMCMCFUNCTIONSANDKERNELS_CU_
