#!/bin/bash
CUDAID=0
SPECFILE1="PN_J0633_15asec_grp15.pi"
#SPECFILE1="pn-thin-5-ao17_0.fak"
#SPECFILE2="pn-thin-5-ao17_1.fak"
SPECFILE2="PN_J0633_15asec_bkg.pi"
CHAINFILE=$1
LOGFILE="LogMetro"
NWALK=128
LSTEP=512
i=$2
NCHAINS=$3
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $CHAINFILE $NWALK $LSTEP $i > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
