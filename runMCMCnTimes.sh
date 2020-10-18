#!/bin/bash
CUDAID=1
SPECFILE1="carbatm_test.fak"
CHAINFILE="carbatm_test"
LOGFILE="carbatm_log"
NWALK=128
LSTEP=$1
i=$2
NCHAINS=$3
emin=$4
emax=$5
let NCHAINS=NCHAINS+i
printf "DeviceID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $SPECFILE1 $emin $emax > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  #./printSpecStat.py $CHAINFILE
  #./plotSpectraFromFile.py $CHAINFILE
  #./printCrdblFromFile.py $CHAINFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
