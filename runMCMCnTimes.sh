#!/bin/bash
CUDAID=0
CHAINFILE="spdeTest6_"
LOGFILE="LogSpdeTest6"
NWALK=128
LSTEP=512
DIM=100
i=$1
NCHAINS=1
SRT=-1
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $DIM $SRT > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
