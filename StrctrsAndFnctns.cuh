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
#define THRDS 32
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
  float scale;
  char *dfl;
  float exptm;
  char *name;
  int indx, dim, nwl, nst, ist, isb, *kr, *kuni;
  float dlt, time;
  float *lst, *stn, *uni, *x0, *stt, *xx, *xx0, *xxC, *xx1, *xxCM, *xCM, *xxW, *zz, *wcnst, *dcnst, *smpls, *stat, *ru, *stt1, *q, *stt0, *xxCP, *zr, *zuni, *runi, *sstt1, *stn1, *rr, *sstt;
  float *stps, *smOfChn, *cntrlChnFnctn, *cmSmMtrx, *chnFnctn, *atcrrFnctn, *cmSmAtCrrFnctn, atcTime;
  cufftComplex *ftOfChn;
  int mmm;
  int nph, nbm;
  float *atms, *nnt, *nt, *mmt, *mt, *mstt, *prr, *prr1, *xbnd, *ccnd, *cnd, *bcnst, *pcnst;
  float *sigma;
  int *bnn;
};


__host__ int grid1D ( const int );
__host__ dim3 grid2D ( const int, const int );
__host__ dim3 block2D ();
__host__ __device__ Complex addComplex ( Complex, Complex );
__host__ __device__ Complex scaleComplex ( Complex, float );
__host__ __device__ Complex multiplyComplex ( Complex, Complex );
__host__ __device__ Complex conjugateComplex ( Complex );

__global__ void scaleArray ( const int, const float, float* );
__global__ void constantArray ( const int, const float, float* );
__global__ void sliceArray ( const int, const int, const float*, float* );
__global__ void sliceIntArray ( const int, const int, const int*, int* );
__global__ void insertArray ( const int, const int, const float*, float* );
__global__ void initializeAtRandom ( const int, const int, const float, const float*, const float*, float* );
__global__ void returnStatistic ( const int, const int, const float*, float* );
__global__ void setWalkersAtLast ( const int, const int, const float*, float* );
__global__ void setStatisticAtLast ( const int, const int, const float*, float* );
__global__ void shiftWalkers ( const int, const int, const float*, const float*, float* );
__global__ void addWalkers ( const int, const int, const float*, const float*, float* );
__global__ void returnQ ( const int, const int, const float*, const float*, const float*, float* );
__global__ void returnQM1 ( const int, const int, const float*, const float*, const float*, const float*, float* );
__global__ void returnQM ( const int, const int, const float*, const float*, float* );
__global__ void updateWalkers ( const int, const int, const float*, const float*, const float*, float* );
__global__ void updateStatistic ( const int, const float*, const float*, const float*, float* );
__global__ void saveWalkers ( const int, const int, const int, const float*, float* );
__global__ void saveStatistic ( const int, const int, const float*, float* );
__global__ void mapRandomNumbers ( const int, const int, const int, const float*, float*, int*, float* );
__global__ void permuteWalkers ( const int, const int, const int*, const float*, float* );
__global__ void TestpermuteWalkers ( const int dim, const int nwl, const int *kr, const float *xxC, float *xxCP );
__global__ void substractWalkers ( const int, const int, const float*, const float*, float* );
__global__ void scale2DArray ( const int, const int, const float*, const float*, float* );
__global__ void complexPointwiseMultiplyByConjugateAndScale ( const int, const int, const float, Complex* );
__global__ void testChainFunction ( const int, const int, const int, float*, Complex* );
__global__ void chainFunction ( const int, const int, const int, const int, const float*, float* );
__global__ void normArray ( const int, float* );
__global__ void metropolisPoposal2 ( const int, const int, const int, const float*, const float*, float* );
__global__ void metropolisPoposal3 ( const int, const int, const int, const float*, const float*, const float*, float* );

__global__ void arrayOf2DConditions ( const int, const int, const float*, const float*, float* );
__global__ void arrayOfPriors ( const int, const int, const float*, const float*, float* );
__host__ __device__ int binNumber ( const int, const float, const float, const float );
__global__ void arrayOfBinTimes ( const int, const int, const int, const float*, const float*, float* );
__global__ void arrayOfMultiplicity ( const int, const int, const int, const float, const float*, float* );
__global__ void arrayOfStat ( const int nbm, const float *mt, float *mstt );
__host__ int modelStatistic ( const Cupar *cdp, Chain *chn );
__host__ dim3 grid3D ( const int, const int, const int, const dim3 );
__host__ int readTimesInfo ( const char*, int*, float* );
__host__ int readTimesData ( const char*, const int, float* );
__host__ int modelStatistic1 ( const Cupar*, Chain* );
__host__ int allocateTimes ( Chain* );

__host__ int initializeCuda ( Cupar* );
__host__ int allocateChain ( Chain * );
__host__ int initializeChain ( Cupar*, Chain* );
__host__ int initializeRandomForWalk ( Cupar*, Chain* );
__host__ int initializeRandomForStreach ( Cupar*, Chain* );
__host__ int walkMove ( const Cupar*, Chain* );
__host__ int streachMove ( const Cupar*, Chain* );
__host__ int statistic ( const Cupar*, Chain* );
__host__ int walkUpdate ( const Cupar*, Chain* );
__host__ int streachUpdate ( const Cupar*, Chain* );
__host__ int saveCurrent ( Chain* );
__host__ void readLastFromFile ( const char*, const int, const int, const int, float* );
__host__ void writeChainToFile ( const char*, const int, const int, const int, const int, const float*, const float* );
__host__ int destroyCuda ( const Cupar* );
__host__ void freeChain ( const Chain* );
__host__ void freeTimes ( const Chain* );
__host__ void simpleReadDataFloat ( const char*, float* );
__host__ void simpleReadDataInt ( const char*, int*);
__host__ void simpleWriteDataFloat ( const char*, const int, const float* );
__host__ void simpleWriteDataFloat2D ( const char*, const int, const int, const float* );
__host__ int printMove ( const Chain* );
__host__ int printUpdate ( const Chain* );
__host__ int printMetropolisMove ( const Chain* );
__host__ int printMetropolisUpdate ( const Chain* );
__host__ int averagedAutocorrelationFunction ( Cupar*, Chain* );
__host__ void cumulativeSumOfAutocorrelationFunction ( const int, const float*, float* );
__host__ int chooseWindow ( const int, const float, const float* );
__host__ int initializeRandomForMetropolis ( Cupar *cdp, Chain *chn );
__host__ int metropolisMove ( const Cupar *cdp, Chain *chn );
__host__ int statisticMetropolis ( const Cupar *cdp, Chain *chn );
__host__ int statistic0 ( const Cupar*, Chain* );
__host__ int metropolisUpdate ( const Cupar*, Chain* );

#endif // _STRCTRSANDFNCTNS_CUH_
