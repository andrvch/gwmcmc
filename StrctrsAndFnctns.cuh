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
#define RANK 1
#define NPRS 7 // Number of parameters
#define THRDS 32
#define RANK 1
#define ACONST 2.0f // Goodman-Weare "a" constant
#define BACKIN 1
#define NSPCTR 6
#define NSPCTR11 12
#define NSPCTRCHI NSPCTR
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

struct arrIndx {
    float value;
    int index;
};

struct arrKde {
    float value;
    float bin;
    int index;
};

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
  int indx, dim, nwl, nst, nsm, ist, isb, *kr, *kuni;
  float dlt, time;
  float *lst, *stn, *uni, *x0, *stt, *xx, *xx0, *xxC, *xx1, *xxCM, *xCM, *xxW, *zz, *wcnst, *dcnst, *smpls, *stat, *ru, *stt1, *q, *stt0, *xxCP, *zr, *zuni, *runi, *sstt1, *stn1, *rr, *sstt, *priors, *nhMd, *nhSg;
  float *stps, *smOfChn, *cntrlChnFnctn, *cmSmMtrx, *chnFnctn, *atcrrFnctn, *cmSmAtCrrFnctn, atcTime;
  cufftComplex *ftOfChn;
  int mmm;
  float *atms, *nnt, *nt, *nt1, *numbers, *mmt, *mt, *mstt, *prr, *prr1, *prr0, *xbnd, *ccnd, *cnd, *bcnst, *pcnst;
  float *sigma;
  int *bnn;
  float *didi, *didi0, *dist, *didi1;
  float *didi01, *didi02, *didi03, *didi11, *didi12, *didi13;
  int *kex;
  float *chi, *chi1, *chi0;
  float *chiTwo;
  float *msmp, *vsmp, *hsmp, *stdsmp, *csmp, *sqcsmp;
  arrIndx *obj;
  int Indx;
  float *param, *sortIndx;
  int *sm;
  float *sp, *ssp, *smVls;
  int *smPntr, *smIndx;
  int nkb;
  float *pdf, *ppdf, *skbin, *maxPdf;
  float *kpar, *kbin, *skpdf;
  float *kdeVls, *kdePdf;
  float *hkde, *lkde;
  float *skdePdf;
  arrKde *objkde;
  float *whales;
  int dim1;
};

struct Spectrum {
  char *name;
  char *spcLst[NSPCTR];
  char srcTbl[FLEN_CARD], arfTbl[FLEN_CARD], rmfTbl[FLEN_CARD], bckgrndTbl[FLEN_CARD];
  float lwrNtcdEnrg, hghrNtcdEnrg;
  int nmbrOfChnnls, nmbrOfEnrgChnnls, nmbrOfRmfVls, nmbrOfBns, nmbrOfNtcdBns, nmbrOfiVls, nmbrOfUsdBns, nmbrOfNtcdChnnls;
  float srcExptm, bckgrndExptm;
  float backscal_src, backscal_bkg;
  int *rmfPntrInCsc, *rmfIndxInCsc, *rmfPntr, *rmfIndx;
  float *grpVls, *ntcVls, *bnsbns, *grpng, *grpIgnVls, *iVls, *srcGrp, *ignRmfVls, *ignVls, *srcIgn;
  int *grpIndx, *grpPntr, *grpIgnIndx, *grpIgnPntr, *iPntr, *iIndx, *ignRmfIndx, *ignRmfPntr;
  int *ntcIndx, *ntcPntr, *ignPntr, *ignIndx;
  float *rmfVlsInCsc, *rmfVls, *enrgChnnls, *arfFctrs, *srcCnts, *bckgrndCnts, *lwrChnnlBndrs, *hghrChnnlBndrs, *gdQltChnnls;
  float *crssctns, *absrptnFctrs, *mdlFlxs, *nsFlxs, *plFlxs, *flddMdlFlxs, *flddNSFlxs, *flddPLFlxs, *ntcdChnnls, *chnnlSttstcs, *chiSttstcs, smOfNtcdChnnls;
  float *nsa1Flxs, *nsa2Flxs;
  int nmbrOfgrpIgnVls, nmbrOfignRmfVls;
  int hghrBn, lwrBn, hghrCh, lwrCh;
  float *bkgGrp, *bkgIgn;
  float *chi, *stat;
  float *lwrGrp, *hghrGrp;
};

struct Model {
  int sgFlg = 3; // Xset.xsect = "bcmc"
  const char *abndncsFl = "AngrAbundances.dat"; // Xset.abund = "angr"
  //const char *abndncsFl = "WilmAbundances.dat"; // Xset.abund = "wilm"
  const int atNm[ATNMR] = { 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 24, 26, 27, 28 };
  int *atmcNmbrs;
  float *abndncs;
  const char *rddnngFl = "Green19.dat";
  const int nmbrOfDistBins = 120;
  const int numRedCol = 6;
  float *RedData, *Dist, *EE, *EBV, *errDist, *errEBV;
  const char *rddnngFl1 = "Green15.dat";
  const int nmbrOfDistBins1 = 1000;
  const int numRedCol1 = 2;
  float *RedData1, *Dist1, *EBV1;
  const char *rddnngFl2 = "Sale14.dat";
  float *RedData2, *Dist2, *EBV2;
  const char *rddnngFl3 = "Drimmel03.dat";
  float *RedData3, *Dist3, *EBV3;
  const char *nsaFl = "nsa_spec_B_1e12G.dat";
  //const char *nsaFl = "nsa_spec_B_1e13G.dat";
  int numNsaE = 1000;
  int numNsaT = 14;
  float *nsaDt, *nsaE, *nsaT, *nsaFlxs;
  //const char *nsmaxgFl = "nsmaxg_HB1260ThB00g1438.in";
  //const char *nsmaxgFl = "nsmaxg_HB1226Thm00g1420.in";
  //const char *nsmaxgFl = "nsmaxg_HB1226Thm90g1420.in";
  const char *nsmaxgFl = "nsmaxg_HB1300Thm90g1420.in";
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

__global__ void saveTheWhalesXX ( const int, const int, const int, float*, const int, const int, const float* );
__global__ void saveTheWhalesX ( const int, const int, const int, const int, float*, const int, const float* );

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
__global__ void mapRandomNumbers ( const int, const int, const int, const float*, float*, int*, float*, int* );
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
__host__ int initializeChain ( Cupar*, Chain* );
__host__ int initializeRandomForWalk ( Cupar*, Chain* );
__host__ int initializeRandomForStreach ( Cupar*, Chain* );
__host__ int walkMove ( const Cupar*, Chain* );
__host__ int streachMove ( const Cupar*, Chain* );
__host__ int statistic ( const Cupar*, Chain* );
__host__ int walkUpdate ( const Cupar*, Chain* );
__host__ int streachUpdate ( const Cupar*, Chain*, Model* );
__host__ int saveCurrent ( Chain* );
__host__ void readLastFromFile ( const char*, const int, const int, const int, float* );
__host__ void writeChainToFile ( const char*, const int, const int, const int, const int, const float*, const float*, const float*, const float*, const float* );
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
__host__ int SpecData ( Cupar*, const int, Model*, Spectrum*, Spectrum* );
__host__ int SpecInfo ( const int, Spectrum* );
__host__ int SpecAlloc ( Chain*, Spectrum* );
__host__ int ReadFitsInfo ( const char*, int*, int*, int*, int*, float*, float*, char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] );
__host__ int ReadFitsData ( const int, const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD], const int, const int, const int, float*, float*, float*, float*, float*, float*, int*, int*, float*, float*, float*, float*, const int, float*, int*, int*, float* );
__host__ void FreeModel ( const Model* );

__global__ void BilinearInterpolation ( const int, const int, const int, const int, const float*, const float*, const float*, const int, const int, const float*, const float*, float* );
__global__ void BilinearInterpolationNsmax ( const int, const int, const int, const int, const float*, const float*, const float*, const int, const int, const float*, const float*, float* );
__global__ void LinearInterpolation ( const int, const int, const int, const float*, const float*, const float*, const float*, float*, float* );
__global__ void LinearInterpolationNoErrors ( const int, const int, const int, const float*, const float*, const float*, float*, float* );
__global__ void AssembleArrayOfModelFluxes ( const int, const int, const int, const float, const float, const float*, const float*, const float*, const float*, const float*, float*, const float* );

__host__ int modelStatistic1 ( const Cupar*, const Model*, Chain*, Spectrum*, Spectrum * );
__host__ int modelStatistic0 ( const Cupar*, const Model*, Chain*, Spectrum*, Spectrum * );
__host__ __device__ float PowerLaw ( const float, const float, const float, const float );
__host__ __device__ float IntegrateNsa ( const float, const float, const float, const float );
__host__ __device__ float IntegrateNsmax ( const float, const float, const float, const float );
__host__ __device__ float BlackBody ( const float, const float, const float, const float );
__host__ __device__ float Poisson ( const float, const float, const float );
__host__ __device__ float PoissonWithBackground ( const float, const float, const float, const float, const float, const float, const float );
__host__ __device__ int FindElementIndex ( const float*, const int, const float );
__host__ __device__ float chisquared ( const float, const float, const float, const float );

__global__ void AssembleArrayOfAbsorptionFactors ( const int, const int, const int, const float*, const float*, const int*, const float*, float* );
__global__ void AssembleArrayOfChannelStatistics ( const int, const int, const float, const float, const float, const float, const float*, const float*, const float*, float* );

__host__ void AssembleArrayOfPhotoelectricCrossections ( const int, const int, int, float*, int*, float* );

__global__ void AssembleArrayOfNoticedChannels ( const int, const float, const float, const float*, const float*, const float*, float* );

__host__ int InitializeModel ( Model* );
__host__ void FreeSpec ( const Spectrum* );

extern "C" float photo_ ( float*, float*, int*, int*, int* );
extern "C" float gphoto_ ( float*, float*, int*, int* );

__global__ void arrayOf2DConditions ( const int, const int, const float*, const float*, float* );
__global__ void arrayOfPriors ( const int, const int, const float*, const float*, float* );

__host__ void SimpleReadNsaTable ( const char*, const int, const int, float*, float*, float*, float* );
__host__ void SimpleReadNsmaxgTable ( const char*, const int, const int, float*, float*, float*, float* );
__host__ void SimpleReadReddenningData ( const char*, const int, float*, float*, float*, float*, float* );
__host__ void SimpleReadReddenningDataNoErrors ( const char*, const int, float*, float*, float* );
__host__ int printSpec ( const Spectrum* );

__global__ void returnQ1 ( const int, const int, const float*, const float*, const float*, const float*, const float*, float* );
__global__ void arrayOfPriors1 ( const int, const int, const float*, const float*, const float*, const float*, float* );
__global__ void setPriorAtLast ( const int, const int, const float*, float* );
__global__ void ReverseLinearInterpolationNoErrors ( const int, const int, const int, const float*, const float*, const float*, float* );
__global__ void setDistanceAtLast ( const int, const int nwl, const float *lst, float *didi );
__global__ void chooseLaw ( const int, const int*, const float*, const float*, const float*, float* );
__global__ void mapKex ( const int, const float*, int* );

__host__ __device__ float cstat ( const float d, const float m );
__global__ void arrayOfCStat ( const int nwl, const int nch, const float t, const float *c, const float *f, float *s );
__global__ void arrayOfChiSquareds ( const int nwl, const int nch, const float t, const float *c, const float *f, float *s );

__host__ __device__ float chi2 ( const float, const float );

__global__ void setChiAtLast ( const int dim, const int nwl, const float *lst, float *stt );

__global__ void arrayOfChiSquaredsWithBackground ( const int nwl, const int nch, const float t, const float *c, const float *b, const float scale, const float *f, float *s );
__global__ void arrayOfWStat ( const int nwl, const int nch, const float ts, const float tb, const float backscal_src, const float backscal_bkg, const float *d, const float *b, const float *f, float *s );

__host__ __device__ float wstat ( const float scnts, const float bcnts, const float mdl, const float ts, const float tb, const float backscal_src, const float backscal_bkg );

__global__ void AssembleArrayOfModelFluxes2 ( const int spIndx, const int nwl, const int nmbrOfEnrgChnnls, const float backscal_src, const float backscal_bkg, const float *en, const float *arf, const float *absrptn, const float *wlk, const float *nsa1Flx, float *flx, const float *didi );

__global__ void powWalkers ( const int n, const float c, const float *a, float *d );
__global__ void scaleWalkers ( const int n, const float c, const float *a, float *d );

/*
__host__ int cmp ( const void *a, const void *b );
__host__ int sortChain ( Chain *chn );
*/

__global__ void sortMatrix ( const int nd, const float *a, float *sm );
__global__ void extractParam ( const int dim, const int nd, const int Indx, const float *s, float *a );
__host__ int sortChain ( Cupar *cdp, Chain *chn );

__host__ int cmp ( const void *a, const void *b );
__host__ int sortQ ( Chain *chn );

__global__ void sortIndex ( const int dim, const int nd, const float *a, int *sm, float *sp );
__host__ int sillySort ( Cupar *cdp, Chain *chn );

__global__ void lineSpace ( const int, const int, const float *l, const float *h, float *b );
__global__ void gaussKde1D ( const int, const int nd, const int nb, const int, const float *h, const float *a, const float *b, float *pdf );
__global__ void sortIndexKde ( const int n, const float *b, const float *a, int *si, float *sb, float *sa );

__global__ void lhkde ( const int n, const float *a, const float *b, float *l, float *h  );
__global__ void sortIndexKde ( const int d, const int n, const float *a, const float *b, float *sa, float *sb );
__host__ int cmpKde ( const void *a, const void *b );
__host__ int sortQKde ( Chain *chn );

__host__ int chainMomentsAndKde ( Cupar *cdp, Chain *chn );

__host__ void writeWhalesToFile ( const char *chainname, const int chaninindx, const int dim, const int n, const float *whales );
__host__ void writeSpectraToFile ( const char *name, const Spectrum *spc, const Spectrum *bkg );
__host__ int modelStatistic00 ( const Cupar *cdp, const Model *mdl, Chain *chn, Spectrum *spc, Spectrum *bkg );

__global__ void arrayOfSourceFluxes ( const int Indx, const int nwl, const int n, const float *en, const float *arf, const float *abs, const float *xx, const float *nsFlx, float *flx, const float *dist );
__global__ void arrayOfBackgroundFluxes ( const int Indx, const int nwl, const int n, const float *en, const float *arf, const float *xx, float *flx );
__global__ void combineSourceAndBackground ( const int nwl, const int n, const float scale, float *src, const float *bkg );
__host__ __device__ float gabs ( const float p0, const float p1, const float p2, const float enrgLwr, const float enrgHghr );

__global__ void arrayOfSourceFluxes2 ( const int Indx, const int nwl, const int n, const float *en, const float *arf, const float *abs, const float *xx, const float *nsFlx, float *flx, const float *dist );

__host__ __device__ float gabs1 ( const float p0, const float p1, const float p2, const float en );
__host__ __device__ float integrateNsaWithGabs ( const float flx0, const float flx1, const float en0, const float en1, const float p0, const float p1, const float p2  );

__host__ void writeKdeToFile ( const char *name, const int dim, const int n, const float *bin, const float *kde );

__global__ void arrayOfNSFluxes ( const int Indx, const int nwl, const int n, const float *en, const float *arf, const float *abs, const float *xx, const float *nsFlx, float *flx, const float *dist );
__global__ void arrayOfPLFluxes ( const int Indx, const int nwl, const int n, const float *en, const float *arf, const float *abs, const float *xx, const float *nsFlx, float *flx, const float *dist );

__host__ void readGreenSamples ( const char *flNm, const int numDist, const int numSmpl, float *greenData, float *Dist, float *EE );
__global__ void chooseSample ( const int nDB, const int si, const float *EE, float *EBV );
__global__ void LinearInterpolationFromSamples ( const int nmbrOfWlkrs, const int nmbrOfDistBins, const int si, const float *Dist, const float *EBV, const float *wlkrs, float *dist );
__global__ void chooseDistance ( const int nwl, const int *kex, const float *didi11, float *didi1 );

#endif // _STRCTRSANDFNCTNS_CUH_
