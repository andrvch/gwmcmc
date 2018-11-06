#!/bin/bash
CUDAID=0
#SPECFILE1="j1741_1-8kev-bar-20arcsec.fits"
SPECFILE1="pn_barycen_0.3-2.0.fits"
#SPECFILE1="PNclean_bary1.fits"
#SPECFILE1="pn_barycen_0.3-10.0_cl.fits"
#SPECFILE1="pn_barycen_0.3-2.0_cl.fits"
#SPECFILE1="pn_barycen.fits"
CHAINFILE=$1
LOGFILE="Log"
NWALK=1
LSTEP=4194304
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
