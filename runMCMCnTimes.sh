#!/bin/bash
CUDAID=1
SPECFILE1="PN_J0633_15asec_grp1_0.fak"
SPECFILE2="PN_J0633_15asec_bkg.pi"
CHAINFILE=$1
LOGFILE=$2
NWALK=128
LSTEP=$3
i=$4
NCHAINS=$5
emin=0.4
emax=7.0
let NCHAINS=NCHAINS+i
printf "DeviceID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
