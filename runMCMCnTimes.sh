#!/bin/bash
CUDAID=0
CHAINFILE="psffit128Test001_11123_19165_src123psr_"
LOGFILE="LogPsfFit128Test001_11123_19165_src123psr"
NWALK=128
LSTEP=1024
IMDIM=41
i=$1
NCHAINS=1
PSFFL1="psf_11123_src1_1pix_new.fits.newpsf"
DATAFL1="img_11123_src1_new.fits.newpsf"
PSFFL2="psf_11123_src2_1pix_new.fits.newpsf"
DATAFL2="img_11123_src2_new.fits.newpsf"
PSFFL3="psf_11123_src3_1pix_new.fits.newpsf"
DATAFL3="img_11123_src3_new.fits.newpsf"
#PSFFL4="psf_11123_psr_1pix_new.fits.newpsf"
#DATAFL4="img_11123_psr_new.fits.newpsf"
PSFFL4="psf_19165_src1_1pix_new.fits.newpsf"
DATAFL4="img_19165_src1_new.fits.newpsf"
PSFFL5="psf_19165_src2_1pix_new.fits.newpsf"
DATAFL5="img_19165_src2_new.fits.newpsf"
PSFFL6="psf_19165_src3_1pix_new.fits.newpsf"
DATAFL6="img_19165_src3_new.fits.newpsf"
#PSFFL8="psf_19165_psr_1pix_new.fits.newpsf"
#DATAFL8="img_19165_psr_new.fits.newpsf"
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $IMDIM $PSFFL1 $DATAFL1 $PSFFL2 $DATAFL2 $PSFFL3 $DATAFL3 $PSFFL4 $DATAFL4 $PSFFL5 $DATAFL5 $PSFFL6 $DATAFL6 > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
