#!/bin/bash
CUDAID=0
CHAINFILE="psf_test"
LOGFILE="psf_log"
NWALK=128
LSTEP=2048 #65536 # 16384 65536 131072
i=$1
NSTRS=3
NIMGS=3
NX=40
NY=40
DT1="11123_src1_smooth.fits.psf"
DT2="11123_src2_smooth.fits.psf"
DT3="11123_src3_smooth.fits.psf"
DT4="19165_src1_smooth.fits.psf"
DT5="19165_src2_smooth.fits.psf"
DT6="19165_src3_smooth.fits.psf"
DT7="20876_src1_smooth.fits.psf"
DT8="20876_src2_smooth.fits.psf"
DT9="20876_src3_smooth.fits.psf"
NCHAINS=1
let NCHAINS=NCHAINS+i
printf "ID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $NSTRS $NIMGS $NX $NY $DATAFILE1 $DT1 $DT2 $DT3 $DT4 $DT5 $DT6 $DT7 $DT8 $DT9  > $LOGFILE
  #./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
