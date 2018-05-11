#ifndef _READFITSDATA_CUH_
#define _READFITSDATA_CUH_

#define NSPCTR 2

__host__ void ReadAllTheFitsData ( const char *[], int * );
void ReadFitsInfo ( const char*, int*, int*, int*, float*, float*, char*, char*, char*, char* );
void ReadFitsData ( const char*, const char*, const char*, const char*, const int, const int, const int, float*, float*, float*, float*, int*, int*, float*, float*, float*, float* );

#endif // _READFITSDATA_CUH_
