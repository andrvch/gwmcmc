#ifndef _STRCTRSANDFNCTNS_CUH_
#define _STRCTRSANDFNCTNS_CUH_

#define PIPI 3.14159265359
#define MNS 1.4e0f
#define RNS 1.3e1f
#define PCCM 3.08567802e18f
#define KMCM 1.0e5f
#define KMCMPCCM -13.48935060694014384023e0f
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
#define NPRS 14
#define NHINDX NPRS-1
#define TINDX 0
#define RINDX1 1
#define GRINDX 1
#define DINDX1 2
#define RINDX 5
#define DINDX2 6
#define NELMS 30
#define ATNMR 18
#define NSPCTR 12
#define BACKIN 1
#define NSTAT 3

/* Walker data type */
typedef union wlk3u
{
  struct wlk3s
  {
    float phtnIndx, nrmlztn, lgTmprtr, rds, dstnc, nh, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17;
  } wlk3s;
  float par[NPRS];
} Walker;

/* Walker data type */
typedef union st3u
{
  struct st3s
  {
    float cst, bst, chi;
  } st3s;
  float par[NSTAT];
} Statistic;

/* Complex data type */
typedef float2 Complex;

struct Cuparam
{
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

struct Spectrum
{
  char *spcLst[NSPCTR];
  char srcTbl[FLEN_CARD], arfTbl[FLEN_CARD], rmfTbl[FLEN_CARD], bckgrndTbl[FLEN_CARD];
  float lwrNtcdEnrg, hghrNtcdEnrg;
  int nmbrOfChnnls, nmbrOfEnrgChnnls, nmbrOfRmfVls;
  float srcExptm, bckgrndExptm;
  float backscal_src, backscal_bkg;
  int *rmfPntrInCsc, *rmfIndxInCsc, *rmfPntr, *rmfIndx;
  float *rmfVlsInCsc, *rmfVls, *enrgChnnls, *arfFctrs, *srcCnts, *bckgrndCnts, *lwrChnnlBndrs, *hghrChnnlBndrs, *gdQltChnnls;
  float *crssctns, *absrptnFctrs, *mdlFlxs, *flddMdlFlxs, *ntcdChnnls, *chnnlSttstcs, smOfNtcdChnnls;
  float *nsa1Flxs, *nsa2Flxs;
};

struct Chain
{
  float dlt;
  char *thrdNm;
  int nmbrOfWlkrs, nmbrOfStps, thrdIndx, nmbrOfRndmVls;
  Walker *wlkrs, *prpsdWlkrs, *chnOfWlkrs, strtngWlkr, *rndmWlkr;
  float *sttstcs, *prpsdSttstcs, *chnOfSttstcs, *zRndmVls, *prrs, *prpsdPrrs, *chnOfPrrs, *nhMd, *nhSg, *rndmVls, *chnFnctn, *atCrrFnctn, *cmSmAtCrrFnctn, *lstWlkrsAndSttstcs, atcTime;
  float elapsedTime, cufftElapsedTime;
};

struct Model
{
  int sgFlg = 3; // Xset.xsect = "bcmc"
  const char *abndncsFl = "AngrAbundances.dat"; // Xset.abund = "angr"
  const int atNm[ATNMR] = { 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 24, 26, 27, 28 };
  int *atmcNmbrs;
  float *abndncs;
  const char *rddnngFl = "reddeningJ0633.dat";
  const int nmbrOfDistBins = 442;
  const int numRedCol = 4;
  float *RedData, *Dist, *EBV, *errDist, *errEBV;
  const char *nsaFl = "nsa_spec_B_1e12G.dat";
  int numNsaE = 1000;
  int numNsaT = 14;
  float *nsaDt, *nsaE, *nsaT, *nsaFlxs;
  const char *nsmaxgFl = "nsmaxg_HB1260ThB00g1438.in";
  int numNsmaxgE = 117;
  int numNsmaxgT = 14;
  float *nsmaxgDt, *nsmaxgE, *nsmaxgT, *nsmaxgFlxs;
};

/* Functions */
extern "C" float photo_ ( float*, float*, int*, int*, int* );
extern "C" float gphoto_ ( float*, float*, int*, int* );
__host__ int Blocks ( const int );
__host__ dim3 Grid ( const int, const int );
__host__ __device__ Walker AddWalkers ( Walker, Walker );
__host__ __device__ Walker ScaleWalker ( Walker, float );
__host__ __device__ Complex AddComplex ( Complex, Complex );
__host__ __device__ Complex ScaleComplex ( Complex, float );
__host__ __device__ Complex MultiplyComplex ( Complex, Complex );
__host__ __device__ Complex ConjugateComplex ( Complex );
__host__ __device__ int PriorCondition ( const Walker );
__host__ __device__ float PriorStatistic ( const Walker, const int, const float, const float );
__host__ __device__ float GaussianAbsorption ( const float, const float, const float, const float );
__host__ __device__ float PowerLaw ( const float, const float, const float, const float );
__host__ __device__ float IntegrateNsa ( const float, const float, const float, const float );
__host__ __device__ float BlackBody ( const float, const float, const float, const float );
__host__ __device__ float Poisson ( const float, const float, const float );
__host__ __device__ float PoissonWithBackground ( const float, const float, const float, const float, const float, const float, const float );
__host__ __device__ int FindElementIndex ( const float*, const int, const float );
__host__ void AssembleArrayOfPhotoelectricCrossections ( const int, const int, int, float*, int*, float* );
__host__ void ReadLastPositionOfWalkersFromFile ( const char*, const int, const int, float* );
__host__ void WriteChainToFile ( const char*, const int, const int, const int, const Walker*, const float*, const float* );
__host__ void SimpleReadNsaTable ( const char*, const int, const int, float*, float*, float*, float* );
__host__ void SimpleReadNsmaxgTable ( const char*, const int, const int, float*, float*, float*, float* );
__host__ void SimpleReadReddenningData ( const char*, const int, float*, float*, float*, float*, float* );
__host__ void SimpleReadDataFloat ( const char*, float* );
__host__ void SimpleReadDataInt ( const char*, int* );
__host__ void SimpleWriteDataFloat ( const char*, const int, const float* );
__host__ void SimpleWriteDataFloat2D ( const char*, const int, const int, const float* );
__host__ void AutocorrelationFunctionAveraged ( cufftResult_t, cublasStatus_t, cublasHandle_t, cufftHandle, const int, const int, const float*, float* );
__host__ void CumulativeSumOfAutocorrelationFunction ( const int, const float*, float* );
__host__ int ChooseWindow ( const int, const float, const float* );
__host__ void FreeSpec ( const Spectrum* );
__host__ void FreeChain ( const Chain* );
__host__ void FreeModel ( const Model* );
__host__ void DestroyAllTheCudaStaff ( const Cuparam* );
__host__ int InitializeCuda ( Cuparam* );
__host__ int InitializeModel ( Model *mdl );
__host__ int InitializeChain ( Cuparam*, const float*, Chain* );
__host__ int ReadFitsInfo ( const char*, int*, int*, int*, float*, float*, char*, char*, char*, char* );
__host__ int ReadFitsData ( const int, const char*, const char*, const char*, const char*, const int, const int, const int, float*, float*, float*, float*, float*, float*, int*, int*, float*, float*, float*, float* );
__host__ int Stat ( const int, Spectrum );
__host__ int SumUpStat ( Cuparam*, const float, const int, float*, const Spectrum );
__host__ int FoldModel ( Cuparam*, const int, Spectrum );
__host__ int ModelFluxes ( const Model*, const int, const Walker*, Spectrum );
__host__ int InitAtRandom ( Cuparam*, Chain* );
__host__ int InitFromLast ( Chain* );
__host__ int Priors ( const Model*, const int, const Walker*, float*, float*, float* );
__host__ int Propose ( const int, const int, Chain* );
__host__ int Update ( const int, const int, Chain* );
__host__ int ToChain ( const int, Chain* );
__host__ int SpecInfo ( const char*[], const int, Spectrum* );
__host__ int SpecAlloc ( Chain*, Spectrum* );
__host__ int SpecData ( Cuparam*, const int, Model*, Spectrum* );

/* Kernels */
__global__ void AssembleArrayOfRandomWalkers ( const int, const float*, Walker* );
__global__ void InitializeWalkersAtRandom ( const int, const float, Walker, Walker*, Walker*, float* );
__global__ void InitializeWalkersAndStatisticsFromLastChain ( const int, const float*, Walker*, float*, float* );
__global__ void WriteWalkersAndStatisticsToChain ( const int, const int, const Walker*, const float*, const float*, Walker*, float*, float* );
__global__ void AssembleArrayOfPriors ( const int, const Walker*, const float*, const float*, float* );
__global__ void AssembleArrayOfAbsorptionFactors ( const int, const int, const int, const float*, const float*, const int*, const Walker*, float* );
__global__ void AssembleArrayOfModelFluxes ( const int, const int, const int, const float, const float, const float*, const float*, const float*, const Walker*, const float*, const float*, float* );
__global__ void AssembleArrayOfNoticedChannels ( const int, const float, const float, const float*, const float*, const float*, float* );
__global__ void AssembleArrayOfChannelStatistics ( const int, const int, const float, const float, const float, const float, const float*, const float*, const float*, float * );
__global__ void GenerateProposal ( const int, const int, const int, const Walker*, const float*, float*, Walker*, float* );
__global__ void UpdateWalkers ( const int, const int, const int, const Walker*, const float*, const float*, const float*, const float*, Walker*, float*, float* );
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
