#!/bin/bash
CUDAID=0
#DATAFILE="pn_barycen_0.3-10.0_cl.fits"
#SPECFILE1="pn_barycen_0.3-2.0.fits"
#SPECFILE1="PNclean_bary1.fits"
#SPECFILE1="pn_barycen_0.3-10.0_cl.fits"
#SPECFILE1="pn_barycen_0.3-2.0_cl.fits"
#SPECFILE1="pn_barycen.fits"
SPECFILE1="PN_J0633_15asec_grp15.pi"
SPECFILE2="PN_J0633_15asec_bkg.pi"
CHAINFILE=$1
LOGFILE="LogMetro"
NWALK=$2
LSTEP=$3
i=$4
NCHAINS=$5
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
