#!/bin/bash

CUDAID=0
SPECFILE=$1
CHAINDIR=$2
if [ ! -d $CHAINDIR  ]; then
    mkdir $CHAINDIR
fi
LOGFILE="Log"
if [ -f $CHAINDIR/$LOGFILE ]; then
   rm $CHAINDIR/$LOGFILE
fi
CHAIN="Chain"

NWALK=128
LSTEP=1024

i=$3
NCHAINS=$4
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
    ./runGwMcmcOnCuda $CUDAID $SPECFILE $CHAINDIR/$CHAIN $NWALK $LSTEP $i >> $CHAINDIR/$LOGFILE
    let i=i+1
    let TOTAL=i*LSTEP
    printf "$TOTAL"
    printf ">"
done
printf "Stop"
printf "\n"
