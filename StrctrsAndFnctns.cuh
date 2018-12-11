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
#define BACKIN 1
#define NSPCTR 12
#define ATNMR 18

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
  float *atms, *nnt, *nt, *nt1, *numbers, *mmt, *mt, *mstt, *prr, *prr1, *xbnd, *ccnd, *cnd, *bcnst, *pcnst;
  float *sigma;
  int *bnn;
};

struct Spectrum {
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

struct Model
{
  int sgFlg = 3; // Xset.xsect = "bcmc"
  const char *abndncsFl = "AngrAbundances.dat"; // Xset.abund = "angr"
  //const char *abndncsFl = "WilmAbundances.dat"; // Xset.abund = "wilm"
  const int atNm[ATNMR] = { 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 24, 26, 27, 28 };
  int *atmcNmbrs;
  float *abndncs;
  const char *rddnngFl = "reddeningJ0633.dat";
  const int nmbrOfDistBins = 442;
  const int numRedCol = 4;
  float *RedData, *Dist, *EBV, *errDist, *errEBV;
  const char *rddnngFl1 = "Green15.dat";
  const int nmbrOfDistBins1 = 1000;
  const int numRedCol1 = 2;
  float *RedData1, *Dist1, *EBV1;
  const char *nsaFl = "nsa_spec_B_1e12G.dat";
  int numNsaE = 1000;
  int numNsaT = 14;
  float *nsaDt, *nsaE, *nsaT, *nsaFlxs;
  //const char *nsmaxgFl = "nsmaxg_HB1260ThB00g1438.in";
  //const char *nsmaxgFl = "nsmaxg_HB1226Thm00g1420.in";
  const char *nsmaxgFl = "nsmaxg_HB1226Thm90g1420.in";
  //const char *nsmaxgFl = "nsmaxg_HB1300Thm90g1420.in";
  //const char *nsmaxgFl = "nsmaxg_HB1300Thm00g1420.in";
  int numNsmaxgE = 117;
  int numNsmaxgT = 14;
  float *nsmaxgDt, *nsmaxgE, *nsmaxgT, *nsmaxgFlxs;
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
__global__ void setWalkersAtLast ( const int, const int, const int, const float*, float* );
__global__ void setStatisticAtLast ( const int, const int, const int, const float*, float* );
__global__ void setNumbersAtLast ( const int, const int, const int, const float*, float* );
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
__global__ void saveNumbers ( const int, const int, const int, const float*, float* );
__global__ void updateNumbers ( const int, const int, const float*, const float*, const float*, float* );

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
__host__ void readLastFromFile ( const char*, const int, const int, const int, const int, float* );
__host__ void writeChainToFile ( const char*, const int, const int, const int, const int, const int, const float*, const float*, const float* );
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

__host__ int ReadFitsInfo ( const char *spcFl, int *nmbrOfEnrgChnnls, int *nmbrOfChnnls, int *nmbrOfRmfVls, float *srcExptm, float *bckgrndExptm, char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] );
__host__ int ReadFitsData ( const int verbose, const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD], const int nmbrOfEnrgChnnls, const int nmbrOfChnnls, const int nmbrOfRmfVls, float *backscal_src, float *backscal_bkg, float *srcCnts, float *bckgrndCnts, float *arfFctrs, float *rmfVlsInCsc, int *rmfIndxInCsc, int *rmfPntrInCsc, float *gdQltChnnls, float *lwrChnnlBndrs, float *hghrChnnlBndrs, float *enrgChnnls );
__host__ int SpecAlloc ( Chain *chn, Spectrum *spc );
__host__ int SpecData ( Cupar *cdp, const int verbose, Model *mdl, Spectrum *spc );
__host__ int SpecInfo ( const char *spcLst[NSPCTR], const int verbose, Spectrum *spc );

#endif // _STRCTRSANDFNCTNS_CUH_
