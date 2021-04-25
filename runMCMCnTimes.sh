#!/bin/bash
CUDAID=0
CHAINFILE="psffit102_11123_src123_"
LOGFILE="LogPsfFit102_11123_src123"
NWALK=128
LSTEP=8192
IMDIM=41
i=$1
NCHAINS=1
PSFFL1="psf_11123_src3_1pix_new.fits.newpsf"
DATAFL1="img_11123_src3_new.fits.newpsf"
PSFFL2="psf_11123_src2_1pix_new.fits.newpsf"
DATAFL2="img_11123_src2_new.fits.newpsf"
PSFFL3="psf_11123_src1_1pix_new.fits.newpsf"
DATAFL3="img_11123_src1_new.fits.newpsf"
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $IMDIM $PSFFL1 $DATAFL1 $PSFFL2 $DATAFL2 $PSFFL3 $DATAFL3 > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
