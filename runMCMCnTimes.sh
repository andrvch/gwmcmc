#!/bin/bash
CUDAID=1
#SPECFILE1="PN_psr12asec_grp1_0.fak"
#SPECFILE2="PN_psr12asec_grp1_1.fak"
#SPECFILE3="PN_psr12asec_grp1_2.fak"
SPECFILE1="PN_psr12asec_grp1.pi"
SPECFILE2="M1_psr12asec_grp1.pi"
SPECFILE3="M2_psr12asec_grp1.pi"
CHAINFILE=$1
LOGFILE=$2
NWALK=128
LSTEP=$3
i=$4
NCHAINS=$5
emin=$6
emax=$7
let NCHAINS=NCHAINS+i
printf "DeviceID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  ./printCrdblFromFile.py $CHAINFILE
  #./printSpecStat.py $CHAINFILE
  #./plotSpectraFromFile.py $CHAINFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
