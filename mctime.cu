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

#include "fitsio.h"

#define INF 2e19
#define PI 3.141592654
#define F0 3.36233

struct vectParam {

    float fr;
    float ph;
    float m1;
    float m2;
    float m3;
    float m4;
    float m5;

    __device__ vectParam( float a, float b, float c, float d, float e, float f, float g ) : fr(a), ph(b), m1(c), m2(d), m3(e), m4(f), m5(g) {}

    __device__ float magnitude( void ) { return m1 + m2 + m3 + m4 + m5; }

    __device__ vectParam operator+ (const vectParam& a) {

        return vectParam( fr+a.fr, ph+a.ph, m1+a.m1, m2+a.m2, m3+a.m3 , m4+a.m4, m5+a.m5);
    }

    __device__ vectParam operator* ( const float& z ) {

        return vectParam( z*fr, z*ph, z*m1, z*m2, z*m3, z*m4, z*m5 );
    }
};

void simpleReadDataFloat( FILE *fp, float *data ) {

    float value;

    int i = 0;

    while ( fscanf(fp, "%e", &value) == 1 ) {

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

            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].fr );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].ph );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].m1 );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].m2 );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].m3 );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].m4 );
            fprintf( fp, " %.8E ", Chain[j+i*2*numWalkers].m5 );
            fprintf( fp, " %.8E\n", ChainStat[j+i*2*numWalkers] );
        }
    }
}

/**
 * CUDA Kernel Device code
 */

__global__ void generateT0( const float     *P0,
                            const int        numWalkers,
                            const float     *T0rand,
                                  vectParam *T0 ) {

    int l = threadIdx.x + blockDim.x * blockIdx.x;

    float deltaFr = P0[1] - P0[0];
    float deltaPh = P0[3] - P0[2];
    float deltaM1 = P0[5] - P0[4];
    float deltaM2 = P0[7] - P0[6];
    float deltaM3 = P0[9] - P0[8];
    float deltaM4 = P0[11] - P0[10];
    float deltaM5 = P0[13] - P0[12];

    if ( l < numWalkers ) {

        T0[l].fr = P0[0] + T0rand[l] * deltaFr;
        T0[l].ph = P0[2] + T0rand[l] * deltaPh;
        T0[l].m1 = P0[4] + T0rand[l] * deltaM1;
        T0[l].m2 = P0[6] + T0rand[l] * deltaM2;
        T0[l].m3 = P0[8] + T0rand[l] * deltaM3;
        T0[l].m4 = P0[10] + T0rand[l] * deltaM4;
        T0[l].m5 = P0[12] + T0rand[l] * deltaM5;


    }
}

__global__ void ReadT0( const float     *LastChain,
                        const int        numWalkers,
                              vectParam *T0,
                              int        numPars,
                              float     *cStat0 ) {

    int l = threadIdx.x + blockDim.x * blockIdx.x;

    if ( l < numWalkers ) {

        T0[l].fr  = LastChain[(numPars+1)*l];
        T0[l].ph  = LastChain[(numPars+1)*l+1];
        T0[l].m1  = LastChain[(numPars+1)*l+2];
        T0[l].m2  = LastChain[(numPars+1)*l+3];
        T0[l].m3  = LastChain[(numPars+1)*l+4];
        T0[l].m4  = LastChain[(numPars+1)*l+5];
        T0[l].m5  = LastChain[(numPars+1)*l+6];
        cStat0[l] = LastChain[(numPars+1)*l+7];

    }
}

__device__ float pRior( vectParam T0 ) {

    float prior;

    float Fr = T0.fr*1.E-6 + F0;

    if ( (Fr >= 3.36230)&&(Fr < 3.36240)&&(T0.ph >= 0)&&(T0.ph <= 2*PI)&&(T0.m1 >= 0.0)&&(T0.m2 >= 0.0)&&(T0.m3 >= 0.0)&&(T0.m4 >= 0.0)&&(T0.m5 >= 0.0) ) {

        prior = -logf(Fr); //0.0; // // //

    } else {

        prior = -INF;
    }

    return prior;
}

__device__ float cstatOnDevice( float tms, vectParam par, float Ttot, int N, int m ) {

    float cstat = 0;
    float f, phi;

    f = par.fr*1.E-6 + F0;
    phi = par.ph;

    float timesMod, jt;
    timesMod = fmodf( 2*PI*f*tms+phi, 2*PI );
    jt = 1 + (m/(2*PI))*timesMod;

    float jtFr;
    float jtInt;
    jtFr = modff( jt, &jtInt );

    float jtJt = jt - jtFr;

    float A = par.magnitude() / m;

    if      ( jtJt == 1 ) { cstat = logf(m*A) - A*Ttot/N + logf( par.m1 / m / A ); }
    else if ( jtJt == 2 ) { cstat = logf(m*A) - A*Ttot/N + logf( par.m2 / m / A ); }
    else if ( jtJt == 3 ) { cstat = logf(m*A) - A*Ttot/N + logf( par.m3 / m / A ); }
    else if ( jtJt == 4 ) { cstat = logf(m*A) - A*Ttot/N + logf( par.m4 / m / A ); }
    else if ( jtJt == 5 ) { cstat = logf(m*A) - A*Ttot/N + logf( par.m5 / m / A ); }

    return cstat;
}

__global__ void cstatGPU( const int    numData,
                          const int    numWalkers,
                          const float  Ttot,
                          const int    m,
                          const float *data,
                            vectParam *T0,
                                float *cStatVec,
                                float *OneOne ) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    float dat;
    //vectParam par;

    if ( (i < numData)&&(j < numWalkers) ) {

        dat = data[i];
        //par = T0[j];

        cStatVec[i+j*numData] = cstatOnDevice( dat, T0[j], Ttot, numData, m ) + pRior( T0[j] );
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

       q = cStat01[l] - oldStat;
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
    cublasHandle_t cublasHandle = 0;              // CUBLAS context

    curandGenerator_t gen;

    cusparseStat = cusparseCreate( &cusparseHandle );
    cusparseStat = cusparseCreateMatDescr( &MatDescr );

    cublasStat = cublasCreate(&cublasHandle);

    cusparseSetMatType( MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( MatDescr, CUSPARSE_INDEX_BASE_ZERO );

    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed( gen, 1234ULL );

    fitsfile *fptr;      /* FITS file pointer, defined in fitsio.h */
    char keyword[FLEN_KEYWORD], colname[FLEN_VALUE];
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */
    int hdunum, hdutype, ncols, anynul, dispwidth[1000];
    int firstcol, lastcol = 0, linewidth;
    long nrows;
    long  firstrow=1, firstelem=1;
    int colnum = 1;
    float enullval=0.0;

    FILE *fp;

    float alpha = 1.0;
    float beta  = 0.0;
    int   incxx = 1;
    int   incyy = 1;

    int numWalkers = atoi( argv[4] );
    int numSteps   = atoi( argv[5] );
    int numPrev    = atoi( argv[6] );

    const char *ThrName  = argv[2];
    int numThr     = atoi( argv[3] );

    int numM = 5;
    int numPars = 2 + numM;

    const char *FitsFileName = argv[1];

    fits_open_file(&fptr, FitsFileName, READONLY, &status);
    fits_movabs_hdu(fptr, 2, &hdutype, &status);
    fits_get_num_rows(fptr, &nrows, &status);

    int numData = nrows;
    float frexp = 0.71;
    float Ttot =  9.1917E+04*frexp;

    double *tms0;
    cudaMallocManaged( (void **)&tms0, numData*sizeof(double) );

    float *tms;
    cudaMallocManaged( (void **)&tms, numData*sizeof(float) );

    fits_read_col_dbl(fptr, colnum, firstrow, firstelem, nrows, enullval, tms0, &anynul, &status);
    fits_close_file(fptr, &status);

    for (int i = 0; i < nrows; i++) {

        tms[i] = tms0[i] - tms0[0];
        //printf( " %.10E ", tms[i] );
    }

    float *P0;
    cudaMallocManaged( (void **)&P0, 2*numPars*sizeof(float) );

    P0[0] = 0.;
    P0[1] = 0.5;
    P0[2] = 1.;
    P0[3] = 1.1;
    P0[4] = 0.010;
    P0[5] = 0.015;
    P0[6] = 0.020;
    P0[7] = 0.025;
    P0[8] = 0.030;
    P0[9] = 0.035;
    P0[10] = 0.040;
    P0[11] = 0.045;
    P0[12] = 0.050;
    P0[13] = 0.055;

    float *OneOne;
    cudaMallocManaged( (void **)&OneOne, numData*sizeof(float) );

    float *LastChain;
    cudaMallocManaged( (void **)&LastChain, (1+numPars)*(2*numWalkers)*sizeof(float) );

    if ( numThr > 0  ) {

        char FileLast[256];
        snprintf(FileLast, sizeof(FileLast), "%s%i", ThrName, numThr-1);
        fp = fopen( FileLast, "r" );
        ReadLastChain( fp, LastChain, numPrev, 2*numWalkers, numPars );
        fclose(fp);
    }

    float *cStat0, *cStat01, *cStatVec;
    cudaMallocManaged( (void **)&cStatVec, numData*2*numWalkers*sizeof(float) );
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

    vectParam *Chain;
    cudaMallocManaged( (void **)&Chain, 2*numWalkers*(numSteps+1)*sizeof(vectParam) );

    float *ChainStat;
    cudaMallocManaged( (void **)&ChainStat, 2*numWalkers*(numSteps+1)*sizeof(float) );

    // Set number of threads per block
    int threadsPerBlock = 32;
    int blocksPerThread = ( 2*numWalkers + threadsPerBlock - 1 ) / threadsPerBlock;
    int blocksPerThread_1 = ( numWalkers + threadsPerBlock - 1 ) / threadsPerBlock;

    dim3 dimBlock( threadsPerBlock, threadsPerBlock );
    dim3 dimGrid( (numData + threadsPerBlock - 1) / threadsPerBlock, (2*numWalkers + threadsPerBlock - 1) / threadsPerBlock);
    dim3 dimGrid_1( (numData + threadsPerBlock - 1) / threadsPerBlock, (numWalkers + threadsPerBlock - 1) / threadsPerBlock);

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    curandGenerateUniform( gen, T0rand, 2*numWalkers );

    if ( numThr == 0 ) {
        // **
        generateT0<<<blocksPerThread, threadsPerBlock>>>( P0, 2*numWalkers, T0rand, T0 );
        // **
        cstatGPU<<<dimGrid, dimBlock>>>( numData, 2*numWalkers, Ttot, numM, tms, T0, cStatVec, OneOne );
        // **
        cublasSgemv( cublasHandle, CUBLAS_OP_T, numData, 2*numWalkers, &alpha, cStatVec, numData, OneOne, incxx, &beta, cStat0, incyy );
    } else if ( numThr > 0 ) {

        ReadT0<<<blocksPerThread, threadsPerBlock>>>( LastChain, 2*numWalkers, T0, numPars, cStat0 );
    }

    // **
    writeToChain<<<blocksPerThread, threadsPerBlock>>>( 2*numWalkers, 0, T0, cStat0, Chain, ChainStat );

    int Step = 1;

    while ( Step < numSteps+1 ) {

        for (int SubSet = 0; SubSet < 2; SubSet++) {

            curandGenerateUniform( gen, Zrand, numWalkers );
            curandGenerateUniform( gen, Jrand, numWalkers );
            curandGenerateUniform( gen, Rrand, numWalkers );

            generateT1<<<blocksPerThread_1, threadsPerBlock>>>( T0, Zrand, Jrand, numWalkers, T01, z, k, SubSet );

            cstatGPU<<<dimGrid_1, dimBlock>>>( numData, numWalkers, Ttot, numM, tms, T01, cStatVec, OneOne );
            cublasSgemv( cublasHandle, CUBLAS_OP_T, numData, numWalkers, &alpha, cStatVec, numData, OneOne, incxx, &beta, cStat01, incyy );

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
    snprintf( FileOut, sizeof(FileOut), "%s%i", ThrName, numThr );

    fp = fopen( FileOut, "w" );
    WriteChain( fp, Chain, ChainStat, numWalkers, numSteps );
    fclose(fp);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    cusparseDestroy( cusparseHandle );
    curandDestroyGenerator(gen);

    cudaFree( P0 );
    cudaFree( tms );
    cudaFree( tms0 );
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
    cudaFree( cStatVec );
    cudaFree( LastChain );

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
