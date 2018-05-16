#!/bin/bash
CUDAID=0
SPECFILE1=$1
SPECFILE2=$2
CHAINFILE=$3
LOGFILE="Log"
NWALK=128
LSTEP=512
i=$4
NCHAINS=$5
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runGwMcmcOnCuda $CUDAID $SPECFILE1 $SPECFILE2 $CHAINFILE $NWALK $LSTEP $i > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
