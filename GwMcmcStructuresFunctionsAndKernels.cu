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

#endif // _GWMCMCFUNCTIONSANDKERNELS_CU_
