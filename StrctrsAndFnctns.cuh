#ifndef _STRCTRSANDFNCTNS_CUH_
#define _STRCTRSANDFNCTNS_CUH_

#define PI 3.14159265359e0f
#define LOGPLANCK 26.1787440e0f
#define INF 2e30f
#define INFi -30e0f
#define TLR 1e-25f
#define ALPHA 1e0f
#define BETA  0e0f
#define INCXX 1
#define INCYY 1
#define THRDSPERBLCK 32
#define RANK 1
#define NPRS 2 // Number of parameters
#define ACONST 2.0f // Goodman-Weare "a" constant

typedef float2 Complex;

struct Cupar {
  int dev;
  cudaError_t err = cudaSuccess;
  int runtimeVersion[4], driverVersion[4];
  cudaDeviceProp prop;
  cusparseStatus_t cusparseStat;
  cusparseHandle_t cusparseHandle = 0;
  cusparseMatDescr_t MatDescr = 0;
  cublasStatus_t cublasStat;
  cublasHandle_t cublasHandle = 0;
  curandGenerator_t curandGnrtr, curandGnrtrHst;
  cufftResult_t cufftRes;
  cufftHandle cufftPlan;
  cudaEvent_t start, stop;
};

struct Chain {
  char *name;
  int indx, dim, nwl, nst, ist, isb;
  float dlt, time;
  float *lst, *stn, *uni, *x0, *xx, *xx0, *xxC, *xx1, *xxCM, *xCM, *xxW, *zz, *wcnst, *dcnst;
};

__host__ int initializeCuda ( Cupar* );
__host__ int destroyCuda ( const Cupar* );

__host__ int initializeChain ( Cupar*, Chain* );
__host__ int grid1D ( const int );
__host__ dim3 grid2D ( const int, const int );
__global__ void constantArray ( const int, const float, float* );
__host__ void freeChain ( const Chain* );

__host__ __device__ Walker AddWalkers ( Walker, Walker );
__host__ __device__ Walker ScaleWalker ( Walker, float );
__host__ __device__ float SumOfWalkerComponents ( const Walker );
__host__ __device__ Complex AddComplex ( Complex, Complex );
__host__ __device__ Complex ScaleComplex ( Complex, float );
__host__ __device__ Complex MultiplyComplex ( Complex, Complex );
__host__ __device__ Complex ConjugateComplex ( Complex );
__host__ __device__ int PriorCondition ( const Walker );
__host__ __device__ float PriorStatistic ( const Walker, const int );
__host__ __device__ int FindElementIndex ( const float*, const int, const float );

__host__ void ReadLastPositionOfWalkersFromFile ( const char*, const int, const int, float* );
__host__ void WriteChainToFile ( const char*, const int, const int, const int, const Walker*, const float*, const float* );
__host__ void SimpleReadDataFloat ( const char*, float* );
__host__ void SimpleReadDataInt ( const char*, int* );
__host__ void SimpleWriteDataFloat ( const char*, const int, const float* );
__host__ void SimpleWriteDataFloat2D ( const char*, const int, const int, const float* );

__host__ void AutocorrelationFunctionAveraged ( cufftResult_t, cublasStatus_t, cublasHandle_t, cufftHandle, const int, const int, const float*, float* );
__host__ void CumulativeSumOfAutocorrelationFunction ( const int, const float*, float* );
__host__ int ChooseWindow ( const int, const float, const float* );


__host__ void FreeChain ( const Chain* );
__host__ void DestroyAllTheCudaStaff ( const Cuparam* );

__host__ int InitAtRandom ( Chain* );
__host__ int InitFromLast ( Chain* );
__host__ int Priors ( const int, const Walker*, float* );
__host__ int MetropolisPropose ( const int, const int, Chain* );
__host__ int Propose ( const int, const int, Chain* );
__host__ int Statistics ( const int, const Walker*, float* );
__host__ int Update ( const int, const int, Chain* );
__host__ int MetropolisUpdate ( const int, Chain* );
__host__ int ToChain ( const int, Chain* );
__host__ void proposeWalkMove ( const int, const int, const Cuparam*, Chain* );

__global__ void AssembleArrayOfStatistic ( const int, const Walker*, float* );
__global__ void AssembleArrayOfRandom2DWalkersFromTwoRandomArrays ( const int, const float*, const float*, Walker* );
__global__ void AssembleArrayOfNoticedTimes ( const int, float* );
__global__ void AssembleArrayOfRandomWalkers ( const int, const float*, Walker* );
__global__ void InitializeWalkersAtRandom ( const int, const float, Walker, Walker*, Walker*, float* );
__global__ void InitializeWalkersAndStatisticsFromLastChain ( const int, const float*, Walker*, float*, float* );
__global__ void WriteWalkersAndStatisticsToChain ( const int, const int, const Walker*, const float*, const float*, Walker*, float*, float* );
__global__ void AssembleArrayOfPriors ( const int, const Walker*, const float* );
__global__ void AssembleArrayOfModelFluxes ( const int, const int, const int, const float, const float, const float*, const float*, const float*, const Walker*, const float*, float* );
__global__ void AssembleArrayOfNoticedChannels ( const int, const float, const float, const float*, const float*, const float*, float* );

__global__ void centrilazeWalkers ( const int, const int, const float*, const float*, float* );
__global__ void sumWalkers ( const int, const int, const float*, const float*, float* );
__global__ void divideWalkers ( const int, const int, const int, const float*, float*, float* );
__global__ void GenerateProposal ( const int, const int, const int, const Walker*, const float*, float*, Walker*, float* );
__global__ void GenerateMetropolis ( const int, const int, const int, const Walker*, const Walker*, Walker*, float* );
__global__ void UpdateWalkers ( const int, const int, const int, const Walker*, const float*, const float*, const float*, const float*, Walker*, float*, float* );
__global__ void MetropolisUpdateOfWalkers ( const int, const int, const Walker*, const float*, const float*, const float*, Walker*, float*, float* );

__global__ void ComplexPointwiseMultiplyByConjugateAndScale ( const int, const int, const float, Complex* );
__global__ void ReturnConstantArray ( const int, const float, float* );
__global__ void ReturnChainFunctionTest ( const int, const int, const int, float*, Complex* );
__global__ void ReturnChainFunction ( const int, const int, const int, const Walker*, float* );
__global__ void ReturnCentralChainFunction ( const int, const int, const float*, const float*, float* );
__global__ void NormalizeChain ( const int, float* );
__global__ void MakeMatrix ( const int, const float*, float* );

__global__ void BilinearInterpolation ( const int, const int, const int, const int, const float*, const float*, const float*, const int, const int, const float*, const Walker*, float* );
__global__ void LinearInterpolation ( const int, const int, const int, const float*, const float*, const float*, const Walker*, float*, float* );

#endif // _STRCTRSANDFNCTNS_CUH_
