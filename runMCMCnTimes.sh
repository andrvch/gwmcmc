#!/bin/bash
CUDAID=0
#DATAFILE="pn_barycen_0.3-10.0_cl.fits"
#SPECFILE1="pn_barycen_0.3-2.0.fits"
#SPECFILE1="PNclean_bary1.fits"
#SPECFILE1="pn_barycen_0.3-10.0_cl.fits"
#DATAFILE="pn_barycen_0.3-10.0_cl.fits"
#DATAFILE="pn_barycen_2.0-10.0_cl.fits"
#SPECFILE1="pn_barycen.fits"
#DATAFILE="J1023_ToAs_on.dat"
#DATAFILE="J1023_ToAs_off.dat"
#DATAFILE="pn_barycen_0.15-0.5.fits"
DATAFILE="pn_barycen_1.0-10.0.fits"
CHAINFILE=$1
LOGFILE="LogMetro"
NWALK=1
LSTEP=65536 #2097152 #1048576 #131072 #65536 #16384 #65536 #65536 # 16384 65536 131072
i=$2
NCHAINS=1
NBNS=$3
FR=2.668025
DFR=0.5E-6
let NCHAINS=NCHAINS+i
printf "ID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $DATAFILE $CHAINFILE $NWALK $LSTEP $i $NBNS $FR $DFR > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
