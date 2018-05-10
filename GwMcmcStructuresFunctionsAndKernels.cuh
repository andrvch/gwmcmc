#ifndef _GWMCMCSTRUCTURESFUNCTIONSANDKERNELS_CUH_
#define _GWMCMCSTRUCTURESFUNCTIONSANDKERNELS_CUH_

#define MNS 1.4e0f
#define RNS 1.3e1f
#define PCCM 3.08567802e18f
#define KMCM 1.0e5f
#define KMCMPCCM -13.48935060694014384023e0f
#define INF 2e30f
#define TLR 1e-10f
#define ALPHA 1e0f 
#define BETA  0e0f
#define INCXX 1
#define INCYY 1
#define THRDSPERBLCK 32
#define RANK 1
#define NPRS 6
#define NHINDX 5
#define NSPCTR 2
#define NELMS 30

/* Walker data type */
typedef union wlk3u
{
    struct wlk3s 
    {
        float phtnIndx, nrmlztn, lgTmprtr, rds, dstnc, nh, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17;
    } wlk3s;
    float par[NPRS];
} Walker;

/* Complex data type */
typedef float2 Complex;

/* Functions */
extern "C" float photo_ ( float*, float*, int*, int*, int* );
extern "C" float gphoto_ ( float*, float*, int*, int* );
__host__ __device__ Walker AddWalkers ( Walker, Walker );
__host__ __device__ Walker ScaleWalker ( Walker, float );
__host__ __device__ Complex AddComplex ( Complex, Complex );
__host__ __device__ Complex ScaleComplex ( Complex, float );
__host__ __device__ Complex MultiplyComplex ( Complex, Complex );
__host__ __device__ Complex ConjugateComplex ( Complex );
__host__ __device__ int PriorCondition ( const Walker );
__host__ __device__ float PriorStatistic ( const Walker, const int, const float, const float ); 
__host__ __device__ float PowerLaw ( const float, const float, const float, const float );
__host__ __device__ float BlackBody ( const float, const float, const float, const float );
__host__ __device__ float Poisson ( const float, const float, const float );
__host__ __device__ int FindElementIndex ( const float*, const int, const float );
__host__ void AssembleArrayOfPhotoelectricCrossections ( const int, const int, int, float*, int*, float* );
__host__ void ReadLastPositionOfWalkersFromFile ( const char*, const int, const int, float* );
__host__ void WriteChainToFile ( const char*, const int, const int, const int, const Walker*, const float* );
__host__ void SimpleReadNsaTable ( const char*, const int, const int, float*, float*, float*, float* );
__host__ void SimpleReadReddenningData ( const char*, const int, float*, float*, float*, float*, float* );
__host__ void SimpleReadDataFloat ( const char*, float* ); 
__host__ void SimpleReadDataInt ( const char*, int* );
__host__ void SimpleWriteDataFloat ( const char*, const int, const float* );
__host__ void SimpleWriteDataFloat2D ( const char*, const int, const int, const float* );
__host__ void AutocorrelationFunctionAveraged ( cufftResult_t, cublasStatus_t, cublasHandle_t, cufftHandle, const int, const int, const float*, float* );
__host__ void CumulativeSumOfAutocorrelationFunction ( const int, const float*, float* );
__host__ int ChooseWindow ( const int, const float, const float* );


/* Kernels */
__global__ void InitializeWalkersAtRandom ( const int, const float, Walker, const float*, Walker* );
__global__ void InitializeWalkersAndStatisticsFromLastChain ( const int, const float*, Walker*, float* );
__global__ void WriteWalkersAndStatisticsToChain ( const int, const int, const Walker*, const float*, Walker*, float* );
__global__ void AssembleArrayOfPriors ( const int, const Walker*, const float*, const float*, float* );
__global__ void AssembleArrayOfAbsorptionFactors ( const int, const int, const int, const float*, const float*, const int*, const Walker*, float* );
__global__ void AssembleArrayOfModelFluxes ( const int, const int, const float*, const float*, const float*, const Walker*, float* );
__global__ void AssembleArrayOfNoticedChannels ( const int, const float, const float, const float*, const float*, const float*, float* );
__global__ void AssembleArrayOfChannelStatistics ( const int, const int, const float, const float, const float*, const float*, const float*, float * ); 
__global__ void GenerateProposal ( const int, const int, const int, const Walker*, const float*, float*, Walker* );
__global__ void UpdateWalkers ( const int, const int, const int, const Walker*, const float*, const float*, const float*, const float*, Walker*, float* );
__global__ void ComplexPointwiseMultiplyByConjugateAndScale ( const int, const int, const float, Complex* );
__global__ void ReturnConstantArray ( const int, const float, float* );
__global__ void ReturnChainFunctionTest ( const int, const int, const int, float*, Complex* );
__global__ void ReturnChainFunction ( const int, const int, const int, const Walker*, float* );
__global__ void ReturnCentralChainFunction ( const int, const int, const float*, const float*, float* );
__global__ void NormalizeChain ( const int, float* );
__global__ void MakeMatrix ( const int, const float*, float* );
__global__ void BilinearInterpolation ( const int, const int, const int, const float*, const float*, const float*, const int, const int, const float*, const Walker*, float* );
__global__ void LinearInterpolation ( const int, const int, const int, const float*, const float*, const float*, const Walker*, float*, float* );

#endif // _GWMCMCFUNCTIONSANDKERNELS_CUH_
