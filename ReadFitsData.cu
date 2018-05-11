#ifndef _READFITSDATA_CU_
#define _READFITSDATA_CU_

#include <fitsio.h>
#include "ReadFitsData.cuh"

void ReadAllTheFitsData ( const char *spcFl, Spectrum spec )
{
    char srcTbl[FLEN_CARD], arfTbl[FLEN_CARD], rmfTbl[FLEN_CARD], bckgrndTbl[FLEN_CARD];

    ReadFitsInfo ( spcFl, &spec.nmbrOfEnrgChnnls, &spec.nmbrOfChnnls, &spec.nmbrOfRmfVls, &spec.srcExptm, &spec.bckgrndExptm, srcTbl, arfTbl, rmfTbl, bckgrndTbl );

    cudaMallocManaged ( ( void ** ) &spec.rmfPntrInCsc, ( spec.nmbrOfEnrgChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec.rmfIndxInCsc, spec.nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec.rmfPntr, ( spec.nmbrOfChnnls + 1 ) * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec.rmfIndx, spec.nmbrOfRmfVls * sizeof ( int ) );
    cudaMallocManaged ( ( void ** ) &spec.rmfVlsInCsc, spec.nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.rmfVls, spec.nmbrOfRmfVls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.enrgChnnls, ( spec.nmbrOfEnrgChnnls + 1 ) * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.arfFctrs, spec.nmbrOfEnrgChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.srcCnts, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.bckgrndCnts, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.lwrChnnlBndrs, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.hghrChnnlBndrs, spec.nmbrOfChnnls * sizeof ( float ) );
    cudaMallocManaged ( ( void ** ) &spec.gdQltChnnls, spec.nmbrOfChnnls * sizeof ( float ) );

    ReadFitsData ( srcTbl, arfTbl, rmfTbl, bckgrndTbl, spec.nmbrOfEnrgChnnls, spec.nmbrOfChnnls, spec.nmbrOfRmfVls,
                   spec.srcCnts, spec.bckgrndCnts, spec.arfFctrs, spec.rmfVlsInCsc, spec.rmfIndxInCsc, spec.rmfPntrInCsc, spec.gdQltChnnls, spec.lwrChnnlBndrs, spec.hghrChnnlBndrs, spec.enrgChnnls );

}

void ReadFitsInfo ( const char *spcFl,
                    int *nmbrOfEnrgChnnls, int *nmbrOfChnnls, int *nmbrOfRmfVls, float *srcExptm, float *bckgrndExptm,
                    char srcTbl[FLEN_CARD], char arfTbl[FLEN_CARD], char rmfTbl[FLEN_CARD], char bckgrndTbl[FLEN_CARD] )
{
    fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
    int status = 0, intnull = 0, anynull = 0, colnum;
    char card[FLEN_CARD], colNgr[] = "N_GRP", colNch[] = "N_CHAN";
    float floatnull;

    /* Open Spectrum  */
    snprintf ( srcTbl, sizeof ( card ), "%s%s", spcFl, "[SPECTRUM]" );
    fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TINT, "NAXIS2", nmbrOfChnnls, NULL, &status );
    fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", srcExptm, NULL, &status );
    fits_read_key ( ftsPntr, TSTRING, "ANCRFILE", card, NULL, &status );
    snprintf ( arfTbl, sizeof ( card ), "%s%s", card, "[SPECRESP]" );
    fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
    snprintf ( rmfTbl, sizeof ( card ), "%s%s", card, "[MATRIX]" );
    fits_read_key ( ftsPntr, TSTRING, "BACKFILE", card, NULL, &status );
    snprintf ( bckgrndTbl, sizeof ( card ), "%s%s", card, "[SPECTRUM]" );
    fits_read_key ( ftsPntr, TSTRING, "BACKFILE", card, NULL, &status );

    /* Open Background file */
    fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TFLOAT, "EXPOSURE", bckgrndExptm, NULL, &status );

    /* Open RMF file */
    fits_open_file ( &ftsPntr, rmfTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TINT, "NAXIS2", nmbrOfEnrgChnnls, NULL, &status );

    int *n_grp;
    n_grp = ( int * ) malloc ( *nmbrOfEnrgChnnls * sizeof ( int ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colNgr, &colnum, &status );
    fits_read_col_int ( ftsPntr, colnum, 1, 1, *nmbrOfEnrgChnnls, intnull, n_grp, &anynull, &status );

    int *n_chan_vec;
    n_chan_vec = ( int * ) malloc ( *nmbrOfChnnls * sizeof ( int ) );

    int sum = 0;

    for ( int i = 0; i < *nmbrOfEnrgChnnls; i++ )
    {
        fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
        fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, n_chan_vec, &anynull, &status );

        for ( int j = 0; j < n_grp[i]; j++ )
        {
            sum = sum + n_chan_vec[j];
        }
    }

    *nmbrOfRmfVls = sum;

    free ( n_chan_vec );
    free ( n_grp );
}

void ReadFitsData ( const char srcTbl[FLEN_CARD], const char arfTbl[FLEN_CARD], const char rmfTbl[FLEN_CARD], const char bckgrndTbl[FLEN_CARD],
                    const int nmbrOfEnrgChnnls, const int nmbrOfChnnls, const int nmbrOfRmfVls,
                    float *srcCnts, float *bckgrndCnts, float *arfFctrs, float *rmfVlsInCsc, int *rmfIndxInCsc, int *rmfPntrInCsc,
                    float *gdQltChnnls, float *lwrChnnlBndrs, float *hghrChnnlBndrs, float *enrgChnnls )
{
    fitsfile *ftsPntr;       /* pointer to the FITS file; defined in fitsio.h */
    int status = 0, anynull, colnum, intnull = 0, rep_chan = 6;
    char card[FLEN_CARD], EboundsTable[FLEN_CARD], Telescop[FLEN_CARD];
    char colNgr[]="N_GRP",colNch[]="N_CHAN",colFch[]="F_CHAN",colCounts[]="COUNTS",colSpecResp[]="SPECRESP",colEnLo[]="ENERG_LO",colEnHi[]="ENERG_HI",colMat[]="MATRIX",colEmin[]="E_MIN",colEmax[]="E_MAX";
    float floatnull, backscal_src, backscal_bkg;

    /* Read Spectrum: */
    fits_open_file ( &ftsPntr, srcTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TSTRING, "RESPFILE", card, NULL, &status );
    snprintf ( EboundsTable, sizeof ( EboundsTable ), "%s%s", card, "[EBOUNDS]" );
    fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", &backscal_src, NULL, &status );
    fits_read_key ( ftsPntr, TSTRING, "TELESCOP", Telescop, NULL, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, srcCnts, &anynull, &status );

    /* Read ARF FILE: */
    fits_open_file ( &ftsPntr, arfTbl, READONLY, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colSpecResp, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, arfFctrs, &anynull, &status );

    /* Read Background: */
    fits_open_file ( &ftsPntr, bckgrndTbl, READONLY, &status );
    fits_read_key ( ftsPntr, TFLOAT, "BACKSCAL", &backscal_bkg, NULL, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colCounts, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, bckgrndCnts, &anynull, &status );

    for ( int i = 0; i < nmbrOfChnnls; i++ )
    {
        bckgrndCnts[i] = bckgrndCnts[i] * backscal_src / backscal_bkg;;
    }

    /* Read RMF file */
    fits_open_file ( &ftsPntr, rmfTbl, READONLY, &status );

    float *enelo_vec, *enehi_vec;
    enelo_vec = ( float * ) malloc ( nmbrOfEnrgChnnls * sizeof ( float ) );
    enehi_vec = ( float * ) malloc ( nmbrOfEnrgChnnls * sizeof ( float ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colEnLo, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, enelo_vec, &anynull, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colEnHi, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfEnrgChnnls, &floatnull, enehi_vec, &anynull, &status );

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        enrgChnnls[i] = enelo_vec[i];
    }

    enrgChnnls[nmbrOfEnrgChnnls] = enehi_vec[nmbrOfEnrgChnnls-1];

    int *f_chan_vec, *n_chan_vec;
    f_chan_vec = ( int * ) malloc ( rep_chan * sizeof ( int ) );
    n_chan_vec = ( int * ) malloc ( rep_chan * sizeof ( int ) );

    int *f_chan, *n_chan;
    f_chan = ( int * ) malloc ( rep_chan * nmbrOfEnrgChnnls * sizeof ( int ) );
    n_chan = ( int * ) malloc ( rep_chan * nmbrOfEnrgChnnls * sizeof ( int ) );

    int *n_grp;
    n_grp = ( int * ) malloc ( nmbrOfEnrgChnnls * sizeof ( int ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colNgr, &colnum, &status );
    fits_read_col_int ( ftsPntr, colnum, 1, 1, nmbrOfEnrgChnnls, intnull, n_grp, &anynull, &status );

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        fits_get_colnum ( ftsPntr, CASEINSEN, colNch, &colnum, &status );
        fits_read_col_int ( ftsPntr, colnum, i+1, 1, n_grp[i], intnull, n_chan_vec, &anynull, &status );

        for ( int j = 0; j < rep_chan; j++ )
        {
            n_chan[i*rep_chan+j] = n_chan_vec[j];
        }
    }

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        fits_get_colnum ( ftsPntr, CASEINSEN, colFch, &colnum, &status );
        fits_read_col ( ftsPntr, TINT, colnum, i+1, 1, n_grp[i], &floatnull, f_chan_vec, &anynull, &status );

        for ( int j = 0; j < rep_chan; j++ )
        {
            f_chan[i*rep_chan+j] = f_chan_vec[j];
        }
    }

    int sum = 0;
    rmfPntrInCsc[0] = 0;

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        for ( int j = 0; j < n_grp[i]; j++ )
        {
            sum = sum + n_chan[rep_chan*i+j];
        }
        rmfPntrInCsc[i+1] = sum;
    }

    int m = 0;

    if ( nmbrOfChnnls == 4096 )
    {
        for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
        {
            for ( int j = 0; j < n_grp[i]; j++ )
            {
                for ( int k = f_chan[rep_chan*i+j] ; k < f_chan[rep_chan*i+j] + n_chan[rep_chan*i+j]; k++ )
                {
                    rmfIndxInCsc[m] = k;
                    m = m + 1;
                }
            }
        }
    }
    else if ( nmbrOfChnnls == 1024 )
    {
        for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
        {
            for ( int j = 0; j < n_grp[i]; j++ )
            {
                for ( int k = f_chan[rep_chan*i+j] - 1; k < f_chan[rep_chan*i+j] - 1 + n_chan[rep_chan*i+j]; k++ )
                {
                    rmfIndxInCsc[m] = k;
                    m = m + 1;
                }
            }
        }
    }

    float *rmf_vec;
    rmf_vec = ( float * ) malloc ( nmbrOfChnnls * sizeof ( float ) );

    fits_get_colnum ( ftsPntr, CASEINSEN, colMat, &colnum, &status );

    m = 0;

    for ( int i = 0; i < nmbrOfEnrgChnnls; i++ )
    {
        sum = rmfPntrInCsc[i+1] - rmfPntrInCsc[i];

        fits_read_col ( ftsPntr, TFLOAT, colnum, i+1, 1, sum, &floatnull, rmf_vec, &anynull, &status );

        for ( int k = 0; k < sum; k++ )
        {
            rmfVlsInCsc[m] = rmf_vec[k];
            //printf ( "%.8E\n", rmfVlsInCsc[m] );
            m = m + 1;
        }
    }

    /* Read Ebounds Table: */
    fits_open_file ( &ftsPntr, EboundsTable, READONLY, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colEmin, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, lwrChnnlBndrs, &anynull, &status );
    fits_get_colnum ( ftsPntr, CASEINSEN, colEmax, &colnum, &status );
    fits_read_col ( ftsPntr, TFLOAT, colnum, 1, 1, nmbrOfChnnls, &floatnull, hghrChnnlBndrs, &anynull, &status );

    free ( enelo_vec );
    free ( enehi_vec );
    free ( rmf_vec );
    free ( f_chan_vec );
    free ( n_chan_vec );
    free ( n_chan );
    free ( f_chan );
    free ( n_grp );
}

#endif // _READFITSDATA_CU_
