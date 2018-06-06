#!/bin/bash
CUDAID=0
SPECFILE1="PN_J0633_15asec_grp15.pi"
SPECFILE2="PN_J0633_15asec_bkg.pi"
SPECFILE3="M1_J0633_15asec_grp15.pi"
SPECFILE4="M1_J0633_bkg.pi"
CHAINFILE=$1
LOGFILE="Log"
<<<<<<< HEAD
NWALK=256
LSTEP=8192
=======
NWALK=128
LSTEP=512
>>>>>>> two_spectra+background
i=$2
NCHAINS=$3
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runGwMcmcOnCuda $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $SPECFILE4 $CHAINFILE $NWALK $LSTEP $i > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
