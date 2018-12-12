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
#define NPRS 10 // Number of parameters
#define THRDS 32
#define RANK 1
#define ACONST 1.5f // Goodman-Weare "a" constant
#define BACKIN 1
#define NSPCTR 4
#define ATNMR 18
#define NELMS 30
#define MNS 1.4e0f
#define RNS 1.3e1f
#define PCCM 3.08567802e18f
#define KMCM 1.0e5f
#define KMCMPCCM -13.48935060694014384023e0f
#define LOGPLANCK 26.1787440e0f
#define NHINDX NPRS-1
#define RINDX1 1
#define TINDX 0
#define GRINDX 1
#define DINDX1 NPRS-2

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
  int indx, dim, nwl, nst, ist, isb, *kr, *kuni;
  float dlt, time;
  float *lst, *stn, *uni, *x0, *stt, *xx, *xx0, *xxC, *xx1, *xxCM, *xCM, *xxW, *zz, *wcnst, *dcnst, *smpls, *stat, *ru, *stt1, *q, *stt0, *xxCP, *zr, *zuni, *runi, *sstt1, *stn1, *rr, *sstt, *priors, *nhMd, *nhSg;
  float *stps, *smOfChn, *cntrlChnFnctn, *cmSmMtrx, *chnFnctn, *atcrrFnctn, *cmSmAtCrrFnctn, atcTime;
  cufftComplex *ftOfChn;
  int mmm;
  float *atms, *nnt, *nt, *nt1, *numbers, *mmt, *mt, *mstt, *prr, *prr1, *prr0, *xbnd, *ccnd, *cnd, *bcnst, *pcnst;
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
  const char *nsmaxgFl = "nsmaxg_HB1226Thm00g1420.in";
  //const char *nsmaxgFl = "nsmaxg_HB1226Thm90g1420.in";
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
__global__ void setWalkersAtLast ( const int, const int, const float*, float* );
__global__ void setStatisticAtLast ( const int, const int, const float*, float* );
__global__ void shiftWalkers ( const int, const int, const float*, const float*, float* );
__global__ void addWalkers ( const int, const int, const float*, const float*, float* );
__global__ void returnQ ( const int, const int, const float*, const float*, const float*, float* );
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

__host__ int initializeCuda ( Cupar* );
__host__ int allocateChain ( Chain * );
__host__ int initializeChain ( Cupar*, Chain*, Model *mdl, Spectrum *spc );
__host__ int initializeRandomForWalk ( Cupar*, Chain* );
__host__ int initializeRandomForStreach ( Cupar*, Chain* );
__host__ int walkMove ( const Cupar*, Chain* );
__host__ int streachMove ( const Cupar*, Chain* );
__host__ int statistic ( const Cupar*, Chain* );
__host__ int walkUpdate ( const Cupar*, Chain* );
__host__ int streachUpdate ( const Cupar*, Chain* );
__host__ int saveCurrent ( Chain* );
__host__ void readLastFromFile ( const char*, const int, const int, const int, float* );
__host__ void writeChainToFile ( const char*, const int, const int, const int, const int, const float*, const float*, const float* );
__host__ int destroyCuda ( const Cupar* );
__host__ void freeChain ( const Chain* );
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

__host__ int SpecData ( Cupar *cdp, const int verbose, Model *mdl, Spectrum *spc );
__host__ int SpecInfo ( const char *spcLst[NSPCTR], const int verbose, Spectrum *spc );
__host__ int SpecAlloc ( Chain *chn, Spectrum *spc );
__host__ int ReadFitsInfo ( const char *spcFl, int *nmbrOfEnrgChnnls, int *nmbrOfChnnls, int *nmbrOfRmfVls, float *srcExptm, float *bckgrndExptm, char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] );
__host__ int ReadFitsData ( const int verbose, const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD], const int nmbrOfEnrgChnnls, const int nmbrOfChnnls, const int nmbrOfRmfVls, float *backscal_src, float *backscal_bkg, float *srcCnts, float *bckgrndCnts, float *arfFctrs, float *rmfVlsInCsc, int *rmfIndxInCsc, int *rmfPntrInCsc, float *gdQltChnnls, float *lwrChnnlBndrs, float *hghrChnnlBndrs, float *enrgChnnls );

__host__ void FreeModel ( const Model *mdl );
__global__ void BilinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int tIndx, const int grIndx, const float *data, const float *xin, const float *yin, const int M1, const int M2, const float *enrgChnnls, const float *wlkrs, float *mdlFlxs );
__global__ void BilinearInterpolationNsmax ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int tIndx, const int grIndx, const float *data, const float *xin, const float *yin, const int M1, const int M2, const float *enrgChnnls, const float *wlkrs, float *mdlFlxs );
__global__ void LinearInterpolation ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int dIndx, const float *Dist, const float *EBV, const float *errEBV, const float *wlkrs, float *mNh, float *sNh );
__global__ void LinearInterpolationNoErrors ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int dIndx, const float *Dist, const float *EBV, const float *wlkrs, float *mNh, float *sNh );
__global__ void AssembleArrayOfModelFluxes ( const int spIndx, const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const float backscal_src, const float backscal_bkg, const float *en, const float *arf, const float *absrptn, const float *wlk, const float *nsa1Flx, float *flx );
__host__ int modelStatistic1 ( const Cupar *cdp, const Model *mdl, Chain *chn, Spectrum *spc );
__host__ int modelStatistic0 ( const Cupar *cdp, const Model *mdl, Chain *chn, Spectrum *spc );
__host__ __device__ float PowerLaw ( const float phtnIndx, const float nrmlztn, const float enrgLwr, const float enrgHghr );
__host__ __device__ float IntegrateNsa ( const float flx1, const float flx2, const float en1, const float en2 );
__host__ __device__ float IntegrateNsmax ( const float flx1, const float flx2, const float en1, const float en2 );
__host__ __device__ float BlackBody ( const float kT, const float logRtD, const float enrgLwr, const float enrgHghr );
__host__ __device__ float Poisson ( const float scnts, const float mdl, const float ts );
__host__ __device__ float PoissonWithBackground ( const float scnts, const float bcnts, const float mdl, const float ts, const float tb, const float backscal_src, const float backscal_bkg );
__host__ __device__ int FindElementIndex ( const float *xx, const int n, const float x );
__global__ void AssembleArrayOfAbsorptionFactors ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int nmbrOfElmnts, const float *crssctns, const float *abndncs, const int *atmcNmbrs, const float *wlkrs, float *absrptnFctrs );
__global__ void AssembleArrayOfChannelStatistics ( const int nmbrOfWlkrs, const int nmbrOfChnnls, const float srcExptm, const float bckgrndExptm, const float backscal_src, const float backscal_bkg, const float *srcCnts, const float *bckgrndCnts, const float *flddMdlFlxs, float *chnnlSttstcs );
__host__ void AssembleArrayOfPhotoelectricCrossections ( const int nmbrOfEnrgChnnls, const int nmbrOfElmnts, int sgFlag, float *enrgChnnls, int *atmcNmbrs, float *crssctns );
__global__ void AssembleArrayOfNoticedChannels ( const int nmbrOfChnnls, const float lwrNtcdEnrg, const float hghrNtcdEnrg, const float *lwrChnnlBndrs, const float *hghrChnnlBndrs, const float *gdQltChnnls, float *ntcdChnnls );
__host__ int InitializeModel ( Model *mdl );
__host__ void FreeSpec ( const Spectrum *spc );
extern "C" float photo_ ( float*, float*, int*, int*, int* );
extern "C" float gphoto_ ( float*, float*, int*, int* );

__global__ void arrayOf2DConditions ( const int dim, const int nwl, const float *bn, const float *xx, float *cc );
__global__ void arrayOfPriors ( const int dim, const int nwl, const float *cn, const float *xx, float *pr );
__host__ void SimpleReadNsaTable ( const char *flNm, const int numEn, const int numTe, float *data, float *Te, float *En, float *fluxes );
__host__ void SimpleReadNsmaxgTable ( const char *flNm, const int numEn, const int numTe, float *data, float *Te, float *En, float *fluxes );
__host__ void SimpleReadReddenningData ( const char *flNm, const int numDist, float *data, float *Dist, float *EBV, float *errDist, float *errEBV );
__host__ void SimpleReadReddenningDataNoErrors ( const char *flNm, const int numDist, float *data, float *Dist, float *EBV );
__host__ int printSpec ( const Spectrum *spc );
__global__ void returnQ1 ( const int dim, const int n, const float *p1, const float *p0, const float *s1, const float *s0, const float *zr, float *q );
__global__ void arrayOfPriors1 ( const int dim, const int nwl, const float *cn, const float *nhMd, const float *nhSg, const float *xx, float *pr );
__global__ void setPriorAtLast ( const int dim, const int nwl, const float *lst, float *prr );

#endif // _STRCTRSANDFNCTNS_CUH_
