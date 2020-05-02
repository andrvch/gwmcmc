#!/bin/bash
CUDAID=0
CHAINFILE="spdeTest1_"
LOGFILE="LogSpdeTest"
NWALK=128
LSTEP=$1
DS=2
EM=20
EN=2
i=$2
NCHAINS=1
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $DS $EM $EN > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
