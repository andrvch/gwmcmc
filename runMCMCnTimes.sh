#!/bin/bash
CUDAID=0
CHAINFILE="psffit01_20876_src1_"
LOGFILE="LogPsfFit01_20876_src1"
NWALK=128
LSTEP=2048
IMDIM=41
i=$1
NCHAINS=1
SRT=0
#PSFFL="psf_19165_psr_1pix_new.fits.newpsf"
#DATAFL="img_19165_psr_new.fits.newpsf"
#PSFFL="psf_19165_src1_1pix_new.fits.newpsf"
#DATAFL="img_19165_src1_new.fits.newpsf"
PSFFL="psf_20876_src1_1pix_new.fits.newpsf"
DATAFL="img_20876_src1_new.fits.newpsf"
#DATAFL="psf_20876_src3_1pix_new.fits.newpsf"
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $IMDIM $SRT $PSFFL $DATAFL > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
