#ifndef _READFITSDATA_CUH_
#define _READFITSDATA_CUH_

#define NSPCTR 2

struct Spectrum
{
    int nmbrOfChnnls, nmbrOfEnrgChnnls, nmbrOfRmfVls;
    float srcExptm, bckgrndExptm;
    int *rmfPntrInCsc, *rmfIndxInCsc, *rmfPntr, *rmfIndx;
    float *rmfVlsInCsc, *rmfVls, *enrgChnnls, *arfFctrs, *srcCnts, *bckgrndCnts, *lwrChnnlBndrs, *hghrChnnlBndrs, *gdQltChnnls;
};

__host__ void ReadAllTheFitsData ( const char*, Spectrum* );
__host__ void ReadFitsInfo ( const char*, int*, int*, int*, float*, float*, char*, char*, char*, char* );
__host__ void ReadFitsData ( const char*, const char*, const char*, const char*, const int, const int, const int, float*, float*, float*, float*, int*, int*, float*, float*, float*, float* );

#endif // _READFITSDATA_CUH_
