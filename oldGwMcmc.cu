// $ nvcc gwMCMCcStatCUDA{...}.cu -o profile_resultGPU -lcurand -lcusparse -lcublas -lcfitsio
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
# include <cuda_runtime.h>
# include <curand.h>
# include <curand_kernel.h>
# include "cusparse_v2.h"
# include "cublas_v2.h"
// For fits files
# include "fitsio.h"

#define INF 2e19f
#define INFi 2e3f
#define PcCm 3.08567802E18
#define KmCm 1.0E5
#define KmCmPcCm -13.48935060694014384023

struct vectParam {

    float nh;
    float ga;
    float np;
    float te;
    float rs;
    float ds;
    
    __device__ vectParam( float a, float b, float c, float d, float e, float f ) : nh(a), ga(b), np(c), te(d), rs(e), ds(f) {}
    
    __device__ vectParam operator+ (const vectParam& a) {
    
        return vectParam( nh+a.nh, ga+a.ga, np+a.np, te+a.te, rs+a.rs, ds+a.ds );
    }
    
    __device__ vectParam operator* ( const float& z ) {
    
        return vectParam( z*nh, z*ga, z*np, z*te, z*rs, z*ds );
    }
};

void readData( FILE  *fp, float *data, float *Te, float *En, float *fluxes, int numTe, int numEn ) {
    
    float value;
    
    int i = 0;
    
    while ( fscanf(fp, "%e", &value) == 1 ) {
            
        data[i] = value;
        i += 1;
    }
    
    for (int j = 0; j < numEn; j++) {
        
        En[j] = data[(j+1)*(numTe+1)];
    }
    
    for (int j = 0; j < numTe; j++) {
        
        Te[j] = data[j+1];
    }
    
    for (int j = 0; j < numEn; j++) {
         for (int i = 0; i < numTe; i++) {
             
             fluxes[j+i*numEn] = log10f(data[(i+1)+(j+1)*(numTe+1)]); 
         }
    }
    
}

void readReddenningData( FILE  *fp, float *data, float *Dist, float *EBV, float *errDist, float *errEBV, int numDist, int numRedCol ) {
    
    float value;
    
    int i = 0;
    
    while ( fscanf(fp, "%e", &value) == 1 ) {
            
        data[i] = value;
        i += 1;
    }
    
    for (int j = 0; j < numDist; j++) {
             
        Dist[j]    = data[4*j];
        EBV[j]     = data[4*j+1];
        errDist[j] = data[4*j+2];
        errEBV[j]  = data[4*j+3];
    }
    
}

void simpleReadDataFloat( FILE *fp, float *data ) {

    float value;
    
    int i = 0;
    
    while ( fscanf(fp, "%e", &value) == 1 ) {
            
        data[i] = value;
        i += 1;
    }  
}

void simpleReadDataInt( FILE *fp, int *data ) {

    int value;
    
    int i = 0;
    
    while ( fscanf(fp, "%i", &value) == 1 ) {
            
        data[i] = value;
        i += 1;
    }  
}

void ReadLastChain( FILE *fp, float *LastChain, int numPrev, int numWalkers, int numPars ) {

    float value;
    
    int i = 0;
    
    while ( fscanf(fp, "%e", &value) == 1 ) {
        
        if ( i > (1+numPars)*numWalkers*numPrev - 1 ) {
        
            LastChain[i - (1+numPars)*numWalkers*numPrev] = value;
        }
        i += 1;
    }  
}

void WriteChain( FILE *fp, vectParam *Chain, float *ChainStat, int numWalkers, int numSteps ) {
     
    for (int i = 0; i < (numSteps+1); i++) {
        for (int j = 0; j < 2*numWalkers; j++) {
            
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].nh );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].ga );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].np );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].te );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].rs );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].ds );
            fprintf( fp, " %.8E\n", ChainStat[j+i*2*numWalkers] );
        }
    }    
}

/**
 * CUDA Kernel Device code
 */

__global__ void generateT0( const float *P0, const int numWalkers, const float *T0rand, vectParam *T0 ) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    
    float deltaNh = P0[1] - P0[0];
    float deltaGa = P0[3] - P0[2]; 
    float deltaNp = P0[5] - P0[4];
    float deltaTe = P0[7] - P0[6];
    float deltaRs = P0[9] - P0[8];
    float deltaDs = P0[11] - P0[10];
    
    if ( l < numWalkers ) {
        
        T0[l].nh = P0[0] + T0rand[l] * deltaNh;
        T0[l].ga = P0[2] + T0rand[l] * deltaGa;
        T0[l].np = P0[4] + T0rand[l] * deltaNp;
        T0[l].te = P0[6] + T0rand[l] * deltaTe;
        T0[l].rs = P0[8] + T0rand[l] * deltaRs;
        T0[l].ds = P0[10] + T0rand[l] * deltaDs;
        
    }
}

__global__ void ReadT0( const float *LastChain, const int numWalkers, vectParam *T0, int numPars, float *cStat0 ) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    
    if ( l < numWalkers ) {
        
        T0[l].nh  = LastChain[(numPars+1)*l];
        T0[l].ga  = LastChain[(numPars+1)*l+1];
        T0[l].np  = LastChain[(numPars+1)*l+2];
        T0[l].te  = LastChain[(numPars+1)*l+3];
        T0[l].rs  = LastChain[(numPars+1)*l+4];
        T0[l].ds  = LastChain[(numPars+1)*l+5];
        cStat0[l] = LastChain[(numPars+1)*l+6];
    }
}

__device__ int locateOnDevice( float *xx, int n, float x ) {

    int ju, jm, jl, jres;
    
    jl = 0;
    ju = n;
    
    while ( ju - jl > 1 ) {
    
        jm = floorf( 0.5 * (ju + jl) );
        
        if ( x >= xx[jm] ) {
            
            jl = jm;
        } else {
            
            ju = jm;
        }
    }
    jres = jl;
    if ( x == xx[0] ) jres = 0;
    if ( x >= xx[n-1] ) jres = n-1;
    return jres;
}

__device__ float cstatOnDevice( float dat, float bac, float mod, float exptime ) {
    
    float cstat = 0;
    
    float d = sqrtf( powf(2*exptime*mod - dat - bac, 2.) + 8*exptime*bac*mod );
    float f = ( dat + bac - 2*exptime*mod + d ) / ( 4*exptime ); 
    
    if ( (dat != 0)&&(bac != 0)&&(mod != 0) ) { 
        
        cstat = 2. * ( mod*exptime + 2*exptime*f - dat*logf(exptime*mod+exptime*f) - bac*logf(exptime*f) - dat*(1 - logf(dat)) - bac*(1-logf(bac)) );
    
    } else if ( (dat == 0)&&(bac != 0)&&(mod != 0) ) {
        
        cstat = mod*exptime - bac*logf(0.5);
    } else if ( (dat != 0)&&(bac == 0)&&(mod != 0)&&(mod < dat/2/exptime) ) {
        
        cstat = - mod*exptime - dat*logf(0.5);
    } else if ( (dat != 0)&&(bac == 0)&&(mod != 0)&&(mod >= dat/2/exptime) ) {
    
        cstat = mod*exptime + dat*(logf(dat) - logf(exptime*mod) - 1);
    } else {
    
        cstat = 0;
    }
    
    return cstat;
}

__device__ float pRior( vectParam T0, float mNh, float sNh ) {
    
    float prior;
    
    if ( (T0.rs>=2.)&&(T0.rs<20.)&&(T0.te >= 5.5)&&(T0.te < 6.9)&&(T0.nh >= 0)&&(T0.nh < 1.1)&&(T0.ga >= 0.01)&&(T0.ga < 5.5)&&(T0.np >= -10.)&&(T0.np < 3.) ) {
        
        prior = powf((T0.nh - mNh), 2) / powf(sNh, 2);
    
    } else {
        
        prior = INF;
    }
    
    return prior;
}

__global__ void bilinearInterpolationKernelGPU( float     *res,
                                                float     *data,
                                                float     *xin, 
                                                float     *yin, 
                                                float     *ene_vec, 
                                                vectParam *T0,
                                                int        M1, 
                                                int        M2, 
                                                int numEnChan, 
                                                int numWalkers,
                                                float gr ) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.y + blockDim.y * blockIdx.y;
    
    float res_tmp1, res_tmp2, res_tmp3;
    float xxout, yyout, xxout1, xxout2;
    
    int v, w;
    
    if ( ( l < numEnChan )&&( k < numWalkers ) ) {
        
        xxout2 = ene_vec[l+1];
        xxout1 = ene_vec[l];
        xxout  = 0.5 * (xxout2 + xxout1) / gr;
        
        float sa = powf(T0[k].rs, 2.);
        float NormD =  - 2*(T0[k].ds);
        float DimConst = 2*KmCmPcCm; 
        
        yyout = T0[k].te;
           
        w = locateOnDevice( xin, M1, xxout );
        v = locateOnDevice( yin, M2, yyout );
           
        float xin1, xin2, yin1, yin2, xinDelta, yinDelta;
           
        xin1 = xin[w];
        xin2 = xin[w+1];
        xinDelta = xin2 - xin1;
        
        yin1 = yin[v];   
        yin2 = yin[v+1]; 
        yinDelta = yin2 - yin1;
        
        xxout = (xxout - xin1) / xinDelta;
        yyout = (yyout - yin1) / yinDelta;
        
        int   ind_x = w;
        float a     = xxout;

        int   ind_y = v; 
        float b     = yyout; 
       
        float d00, d01, d10, d11;
        
        if ( (ind_x   < M1)&&(ind_y   < M2) ) d00 = data[ind_y*M1+ind_x];        else    d00 = 0; //-INF; //
        if ( (ind_x+1 < M1)&&(ind_y   < M2) ) d10 = data[ind_y*M1+ind_x+1];      else    d10 = 0; //-INF; //0; 
        if ( (ind_x   < M1)&&(ind_y+1 < M2) ) d01 = data[(ind_y+1)*M1+ind_x];    else    d01 = 0; //-INF; //0;
        if ( (ind_x+1 < M1)&&(ind_y+1 < M2) ) d11 = data[(ind_y+1)*M1+ind_x+1];  else    d11 = 0; //-INF; //0;
        
        res_tmp1 = a * d10 + (-d00 * a + d00);
        res_tmp2 = a * d11 + (-d01 * a + d01);
        
        res_tmp3 = b * res_tmp2 + (-res_tmp1 * b + res_tmp1);
        res_tmp3 = powf( 10., res_tmp3 + NormD + DimConst );
        res_tmp3 = sa * res_tmp3;
        
        res[l+k*numEnChan] = res_tmp3 * ( xxout2 - xxout1 );
    }
}

__global__ void RedLinearInterpolationKernelGPU( float     *mNh,
                                                 float     *sNh,
                                                 float     *Dist,
                                                 vectParam *T0,
                                                 float     *EBV,
                                                 float     *errEBV,
                                                 int numDist, 
                                                 int numWalkers) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    
    float xxout;
    int v;
    
    if ( l < numWalkers ) {
        
        xxout = powf(10,T0[l].ds);
        v = locateOnDevice( Dist, numDist, xxout );
           
        float xin1, xin2, xinDelta;
           
        xin1 = Dist[v];
        xin2 = Dist[v+1];
        xinDelta = xin2 - xin1;
        
        xxout = (xxout - xin1) / xinDelta;
        
        int   ind_x = v;
        float a     = xxout;

        float dmNh0, dmNh1, dsNh0, dsNh1;
        
        dmNh0 = EBV[ind_x];       
        dmNh1 = EBV[ind_x+1];      
        dsNh0 = errEBV[ind_x];  
        dsNh1 = errEBV[ind_x+1];
        
        float res_mNh, res_sNh;
        res_mNh = a * dmNh1 + (-dmNh0 * a + dmNh0);
        res_sNh = a * dsNh1 + (-dsNh0 * a + dsNh0);
        
        mNh[l] = 0.7*res_mNh;
        sNh[l] = 0.7*res_sNh;
    }
}


__global__ void combineVectorsPointWise( float     *a, 
                                         float     *phabs, 
                                         float     *ene_vec, 
                                         vectParam *T0, 
                                         float     *b, 
                                         float     *c, 
                                         int numEnChan,
                                         int numWalkers ) {
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    float xxout1, xxout2, xxout, cc, pl, nsa;
    
    if ( (i < numEnChan)&&(j < numWalkers) ) {
        
        xxout2 = ene_vec[i+1];
        xxout1 = ene_vec[i];
        
        xxout = 0.5 * (xxout2 + xxout1);
        
        nsa = b[i+j*numEnChan]; 
        pl  = powf(xxout, -T0[j].ga) * powf(10., T0[j].np) * ( xxout2 - xxout1 );
    
        cc = a[i] * powf(phabs[i]/1.49548033E-02, T0[j].nh) * ( nsa + pl ) ; 
        
        c[i+j*numEnChan] = cc;
        
    }
}

__global__ void cstatGPU( const int numChan, 
                          const int numWalkers, 
                          const float  exptime,
                          vectParam *T0,
                                float *mNh,
                                float *sNh, 
                          const float *data,
                          const float *backgr,
                          const float *model,
                                float *cStatVec, 
                                float *OneOne ) {
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    float dat, mod, bac;
    
    if ( (i < numChan)&&(j < numWalkers) ) {
     
        dat = exptime*data[i];
        bac = exptime*backgr[i];
        mod = model[i+j*numChan];
        
        cStatVec[i+j*numChan] = cstatOnDevice( dat, bac, mod, exptime ) + pRior( T0[j], mNh[j], sNh[j] );
        OneOne[i] = 1.;
    }


}

__global__ void writeToChain( const int        numWalkers, 
                              const int        Step, 
                              const vectParam *T0,
                              const float     *cStat0,
                                    vectParam *Chain,
                                    float     *ChainStat ) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    
    if ( l < numWalkers ) {
    
        Chain[l+Step*numWalkers] = T0[l];
        ChainStat[l+Step*numWalkers] = cStat0[l];
    }
}

__global__ void generateT1(       vectParam *T0, 
                            const float     *Zrand, 
                            const float     *Jrand,
                            const int        numWalkers,
                                  vectParam *T01,
                                  float     *z,
                                  int       *k,
                            const int        Group ) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    
    if ( l < numWalkers ) {
            
            float zz = 0.5 * powf(Zrand[l]+1, 2.);
            
            float kkk = Jrand[l] * ( numWalkers - 1 + 0.999999 );
            int kk = (int)truncf( kkk );
            
            z[l] = zz;
            k[l] = kk;
        
            T01[l] = T0[kk+(1-Group)*numWalkers] + ( T0[l+Group*numWalkers] + T0[kk+(1-Group)*numWalkers]*(-1) )*zz;
    }
}

__global__ void newPosition( const int        numWalkers,
                             const vectParam *T01,
                             const float     *cStat01,
                             const float     *Rrand,
                                   vectParam *T0,
                                   float     *cStat0,
                             const int        Group ) {
    
    int l = threadIdx.x + blockDim.x * blockIdx.x;
    
    float q;
    
    if ( l < numWalkers ) {
        
       float oldStat = cStat0[l+Group*numWalkers];
    
       q = - 0.5 * ( cStat01[l] - oldStat );
       q = expf(q);
       
       if ( q >= Rrand[l] ) {
           
           T0[l+Group*numWalkers] = T01[l];
           cStat0[l+Group*numWalkers] = cStat01[l];
       } else {
           
           T0[l+Group*numWalkers] = T0[l+Group*numWalkers];
           cStat0[l+Group*numWalkers] = cStat0[l+Group*numWalkers];
       }
           
    }
}

/**
 * Host main routine
 */
int main( int argc, char *argv[] ){
    
    cudaError_t err = cudaSuccess;
    
    cusparseStatus_t cusparseStat;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t MatDescr = 0;
    
    cublasStatus_t cublasStat;                // CUBLAS functions status
    cublasHandle_t cublasHandle = 0;          // CUBLAS context
    
    curandGenerator_t gen;
    
    cusparseStat = cusparseCreate( &cusparseHandle );
    cusparseStat = cusparseCreateMatDescr( &MatDescr );
    
    cublasStat = cublasCreate(&cublasHandle);

    cusparseSetMatType( MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( MatDescr, CUSPARSE_INDEX_BASE_ZERO );
    
    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed( gen, 1234ULL );
    
    FILE *fp;
    
    float alpha = 1.0;
    float beta  = 0.0;
    int   incxx = 1;
    int   incyy = 1;
    
    const char *ThrName  = argv[1];
    
    int numThr     = atoi( argv[2] );
    int numWalkers = atoi( argv[3] );
    int numSteps   = atoi( argv[4] );
    int numPrev    = atoi( argv[5] );
    
    int numPars = 6;
    
    float mns =  1.4;
    float rns =  13.;
    
    //float sa = powf(rns/3.086e13, 2.);
    float gr = 1./1.21; //sqrtf(1.-2.952*mns/rns);

    int numEn = 1000;
    int numTe = 14;
    
    int numNonZeroRmf = 930580; //924237; //382170;
    int numEnChan = 2067; //1070;
    int numChan = 4096; //1024;
    float exptime = 3.306E4; //1.9E4;
    
    float *P0;
    cudaMallocManaged( (void **)&P0, 2*numPars*sizeof(float) );
    
    P0[0] = 0.01;
    P0[1] = 0.1;
    P0[2] = 1.4999;
    P0[3] = 1.5001;
    P0[4] = -6;
    P0[5] = -5.69;
    P0[6] = 5.7;
    P0[7] = 5.75;
    P0[8] = 13.1;
    P0[9] = 13.2;
    P0[10] = 3.;
    P0[11] = 3.01;
    
    float *OneOne;
    cudaMallocManaged( (void **)&OneOne, numChan*sizeof(float) );
    
    float *LastChain;
    cudaMallocManaged( (void **)&LastChain, (1+numPars)*(2*numWalkers)*sizeof(float) );
    
    if ( numThr > 0  ) {
        
        char FileLast[256];
        snprintf(FileLast, sizeof(FileLast), "%s_%i", ThrName, numThr-1);
        fp = fopen( FileLast, "r" );
        ReadLastChain( fp, LastChain, numPrev, 2*numWalkers, numPars );
        fclose(fp);
    }
    
    int numDist = 442;
    int numRedCol = 4;
    
    float *RedData, *Dist, *EBV, *errDist, *errEBV;
    cudaMallocManaged( (void **)&RedData, numDist*numRedCol*sizeof(float) );
    cudaMallocManaged( (void **)&Dist, numDist*sizeof(float) );
    cudaMallocManaged( (void **)&EBV, numDist*sizeof(float) );
    cudaMallocManaged( (void **)&errDist, numDist*sizeof(float) );
    cudaMallocManaged( (void **)&errEBV, numDist*sizeof(float) );
    
    fp = fopen("reddening.data", "r");
    readReddenningData( fp, RedData, Dist, EBV, errDist, errEBV, numDist, numRedCol );
    fclose(fp);
        
    float *data, *En, *Te, *fluxes;
    cudaMallocManaged( (void **)&data, (numEn+1)*(numTe+1)*sizeof(float) );
    cudaMallocManaged( (void **)&En, numEn*sizeof(float) );
    cudaMallocManaged( (void **)&Te, numTe*sizeof(float) );
    cudaMallocManaged( (void **)&fluxes, numEn*numTe*sizeof(float) );
    
    fp = fopen("nsa_spec_B_1e12G.dat", "r");
    readData( fp, data, Te, En, fluxes, numTe, numEn );
    fclose(fp);
    
    float *rmf_val;
    cudaMallocManaged( (void **)&rmf_val, numNonZeroRmf*sizeof(float) );
    
    int *rmf_ptr, *rmf_ind;
    cudaMallocManaged( (void **)&rmf_ind, numNonZeroRmf*sizeof(int) );
    cudaMallocManaged( (void **)&rmf_ptr, (numChan+1)*sizeof(int) );
    
    float *ene_vec, *arf_vec, *dat_vec, *bak_vec, *phabs_vec;
    cudaMallocManaged( (void **)&ene_vec, (numEnChan+1)*sizeof(float) );
    cudaMallocManaged( (void **)&arf_vec, numEnChan*sizeof(float) );
    cudaMallocManaged( (void **)&dat_vec, numChan*sizeof(float) );
    cudaMallocManaged( (void **)&bak_vec, numChan*sizeof(float) );
    cudaMallocManaged( (void **)&phabs_vec, numChan*sizeof(float) );
    
    fp = fopen( "rmf_val_xmm.dat", "r" );
    simpleReadDataFloat( fp, rmf_val );
    fclose(fp);
    
    fp = fopen( "rmf_ptr_xmm.dat", "r" );
    simpleReadDataInt( fp, rmf_ptr );
    fclose(fp);
    
    fp = fopen( "rmf_ind_xmm.dat", "r" );
    simpleReadDataInt( fp, rmf_ind );
    fclose(fp);
    
    fp = fopen( "ene_vec_xmm.dat", "r" );
    simpleReadDataFloat( fp, ene_vec );
    fclose(fp);
    
    fp = fopen( "arf_vec_xmm.dat", "r" );
    simpleReadDataFloat( fp, arf_vec );
    fclose(fp);
    
    fp = fopen( "phabs_dat_vec_fak_xmm.dat", "r" );
    simpleReadDataFloat( fp, phabs_vec );
    fclose(fp);
    
    fp = fopen( "dat_vec_xmm.dat", "r" );
    simpleReadDataFloat( fp, dat_vec );
    fclose(fp);
    
    fp = fopen( "bak_vec_xmm.dat", "r" );
    simpleReadDataFloat( fp, bak_vec );
    fclose(fp);
    
    float *unFolded, *unFoldedArfCorr, *Folded;
    cudaMallocManaged( (void **)&unFolded, 2*numWalkers*numEnChan*sizeof(float) );
    cudaMallocManaged( (void **)&unFoldedArfCorr, 2*numWalkers*numEnChan*sizeof(float) );
    cudaMallocManaged( (void **)&Folded, 2*numWalkers*numChan*sizeof(float) );
    
    float *cStat0, *cStat01, *cStatVec;
    cudaMallocManaged( (void **)&cStatVec, numChan*2*numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&cStat0, 2*numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&cStat01, numWalkers*sizeof(float) );
    
    float *T0rand, *Zrand, *Rrand, *Jrand, *z; 
    cudaMallocManaged( (void **)&T0rand, 2*numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&Zrand, numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&Rrand, numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&Jrand, numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&z, numWalkers*sizeof(float) );
    
    int *k;
    cudaMallocManaged( (void **)&k, numWalkers*sizeof(int) );
    
    vectParam *T0, *T01;
    cudaMallocManaged( (void **)&T0, 2*numWalkers*sizeof(vectParam) );
    cudaMallocManaged( (void **)&T01, numWalkers*sizeof(vectParam) );
    
    float *mNh, *sNh;
    cudaMallocManaged( (void **)&mNh, 2*numWalkers*sizeof(float) );
    cudaMallocManaged( (void **)&sNh, 2*numWalkers*sizeof(float) );
    
    vectParam *Chain;
    cudaMallocManaged( (void **)&Chain, 2*numWalkers*(numSteps+1)*sizeof(vectParam) );
    
    float *ChainStat;
    cudaMallocManaged( (void **)&ChainStat, 2*numWalkers*(numSteps+1)*sizeof(float) );
    
    // Set number of threads per block
    int threadsPerBlock = 32;
    int blocksPerThread = ( 2*numWalkers + threadsPerBlock - 1 ) / threadsPerBlock;
    int blocksPerThread_1 = ( numWalkers + threadsPerBlock - 1 ) / threadsPerBlock;
    
    dim3 dimBlock( threadsPerBlock, threadsPerBlock );
    dim3 dimGrid( (numEnChan + threadsPerBlock - 1) / threadsPerBlock, (2*numWalkers + threadsPerBlock - 1) / threadsPerBlock);
    dim3 dimGrid_1( (numEnChan + threadsPerBlock - 1) / threadsPerBlock, (numWalkers + threadsPerBlock - 1) / threadsPerBlock);
    dim3 dimGrid_2( (numChan + threadsPerBlock - 1) / threadsPerBlock, (2*numWalkers + threadsPerBlock - 1) / threadsPerBlock);
    dim3 dimGrid_3( (numChan + threadsPerBlock - 1) / threadsPerBlock, (numWalkers + threadsPerBlock - 1) / threadsPerBlock);
    
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    
    if ( numThr == 0 ) {
        
        curandGenerateUniform( gen, T0rand, 2*numWalkers );
        
        generateT0<<<blocksPerThread, threadsPerBlock>>>( P0, 2*numWalkers, T0rand, T0 );
    
        bilinearInterpolationKernelGPU<<<dimGrid, dimBlock>>>( unFolded, fluxes, En, Te, ene_vec, T0, numEn, numTe, numEnChan, 2*numWalkers, gr );
        combineVectorsPointWise<<<dimGrid, dimBlock>>>( arf_vec, phabs_vec, ene_vec, T0, unFolded, unFoldedArfCorr, numEnChan, 2*numWalkers );
    
        cusparseScsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, numChan, 2*numWalkers, numEnChan, numNonZeroRmf, &alpha, MatDescr, rmf_val, rmf_ptr, rmf_ind, unFoldedArfCorr, numEnChan, &beta, Folded, numChan);
        
        RedLinearInterpolationKernelGPU<<<blocksPerThread, threadsPerBlock>>>( mNh, sNh, Dist, T0, EBV, errEBV, numDist, 2*numWalkers);
        
        cstatGPU<<<dimGrid_2, dimBlock>>>( numChan, 2*numWalkers, exptime, T0, mNh, sNh, dat_vec, bak_vec, Folded, cStatVec, OneOne );
        cublasSgemv( cublasHandle, CUBLAS_OP_T, numChan, 2*numWalkers, &alpha, cStatVec, numChan, OneOne, incxx, &beta, cStat0, incyy );
        
    } else if ( numThr > 0 ) {
        
        ReadT0<<<blocksPerThread, threadsPerBlock>>>( LastChain, 2*numWalkers, T0, numPars, cStat0 );
    }
    
    writeToChain<<<blocksPerThread, threadsPerBlock>>>( 2*numWalkers, 0, T0, cStat0, Chain, ChainStat );
    
    int Step = 1; 
    
    while ( Step < numSteps+1 ) {
    
        for (int SubSet = 0; SubSet < 2; SubSet++) {
            
            curandGenerateUniform( gen, Zrand, numWalkers );
            curandGenerateUniform( gen, Jrand, numWalkers );
            curandGenerateUniform( gen, Rrand, numWalkers );
            
            generateT1<<<blocksPerThread_1, threadsPerBlock>>>( T0, Zrand, Jrand, numWalkers, T01, z, k, SubSet );
            
            bilinearInterpolationKernelGPU<<<dimGrid_1, dimBlock>>>( unFolded, fluxes, En, Te, ene_vec, T01, numEn, numTe, numEnChan, numWalkers, gr );
            combineVectorsPointWise<<<dimGrid_1, dimBlock>>>( arf_vec, phabs_vec, ene_vec, T01, unFolded, unFoldedArfCorr, numEnChan, numWalkers );
            
            cusparseScsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, numChan, numWalkers, numEnChan, numNonZeroRmf, &alpha, MatDescr, rmf_val, rmf_ptr, rmf_ind, unFoldedArfCorr, numEnChan, &beta, Folded, numChan );
            
            RedLinearInterpolationKernelGPU<<<blocksPerThread_1, threadsPerBlock>>>( mNh, sNh, Dist, T01, EBV, errEBV, numDist, numWalkers);
            
            cstatGPU<<<dimGrid_3, dimBlock>>>( numChan, numWalkers, exptime, T01, mNh, sNh, dat_vec, bak_vec, Folded, cStatVec, OneOne );
            cublasSgemv( cublasHandle, CUBLAS_OP_T, numChan, numWalkers, &alpha, cStatVec, numChan, OneOne, incxx, &beta, cStat01, incyy );
            
            newPosition<<<blocksPerThread_1, threadsPerBlock>>>( numWalkers, T01, cStat01, Rrand, T0, cStat0, SubSet );
        }
        
        writeToChain<<<blocksPerThread, threadsPerBlock>>>( 2*numWalkers, Step, T0, cStat0, Chain, ChainStat );
        
        Step += 1;
        
    }
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    
    float elapsedTime;
    
    cudaEventElapsedTime( &elapsedTime, start, stop );
    
    printf( "Time to generate: %3.1f ms\n", elapsedTime );
    
    char FileOut[256];
    snprintf( FileOut, sizeof(FileOut), "%s_%i", ThrName, numThr ); 
    
    fp = fopen( FileOut, "w" );
    WriteChain( fp, Chain, ChainStat, numWalkers, numSteps );
    fclose( fp );
    
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    cusparseDestroy( cusparseHandle );
    curandDestroyGenerator( gen );
    
    cudaFree( P0 );
    cudaFree( RedData );
    cudaFree( Dist );
    cudaFree( EBV );
    cudaFree( errDist );
    cudaFree( errEBV );
    cudaFree( data );
    cudaFree( En );
    cudaFree( Te );
    cudaFree( fluxes );
    cudaFree( ene_vec );
    cudaFree( rmf_val );
    cudaFree( rmf_ind );
    cudaFree( rmf_ptr );
    cudaFree( arf_vec );
    cudaFree( phabs_vec );
    cudaFree( dat_vec );
    cudaFree( bak_vec );
    cudaFree( unFolded );
    cudaFree( unFoldedArfCorr );
    cudaFree( Folded );
    cudaFree( cStat0 );
    cudaFree( cStat01 );
    cudaFree( T0rand );
    cudaFree( Zrand );
    cudaFree( Rrand );
    cudaFree( Jrand );
    cudaFree( T0 );
    cudaFree( T01 );
    cudaFree( z );
    cudaFree( k );
    cudaFree( Chain );
    cudaFree( ChainStat );
    cudaFree( OneOne );
    cudaFree( LastChain );
    cudaFree( cStatVec );
    cudaFree( mNh );
    cudaFree( sNh );
    
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess) {
        
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Done\n");
    return 0;
}
