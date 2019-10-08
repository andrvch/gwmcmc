#!/bin/bash
CUDAID=0
#DATAFILE="pn_barycen_0.3-10.0_cl.fits"
#SPECFILE1="pn_barycen_0.3-2.0.fits"
#SPECFILE1="PNclean_bary1.fits"
#SPECFILE1="pn_barycen_0.3-10.0_cl.fits"
DATAFILE="pn_barycen_0.3-10.0_cl.fits"
#SPECFILE1="pn_barycen.fits"
CHAINFILE=$1
LOGFILE="LogMetro"
NWALK=1
LSTEP=524288
i=$2
NCHAINS=1
NBNS=$3
let NCHAINS=NCHAINS+i
printf "ID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $DATAFILE $CHAINFILE $NWALK $LSTEP $i $NBNS > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
