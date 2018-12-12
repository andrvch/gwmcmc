#!/bin/bash
CUDAID=0
SPECFILE1="PN_J0633_15asec_grp15.pi"
SPECFILE2="PN_J0633_15asec_bkg.pi"
CHAINFILE=$1
LOGFILE="LogMetro"
NWALK=128
LSTEP=2048
i=$2
NCHAINS=$3
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE1 $CHAINFILE $NWALK $LSTEP $i > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
