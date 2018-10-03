#!/bin/bash
CUDAID=0
SPECFILE1="PNclean_bary1.fits"
CHAINFILE=$1
LOGFILE="Log"
NWALK=1024
LSTEP=1024
i=$2
NCHAINS=$3
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runGwMcmcOnCuda $CUDAID $SPECFILE1 $CHAINFILE $NWALK $LSTEP $i > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
