#!/bin/bash
CUDAID=0
SPECFILE1="j1741_1-8kev-bar-20arcsec.fits"
CHAINFILE=$1
LOGFILE="Log"
NWALK=2
LSTEP=1
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
