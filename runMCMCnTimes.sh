#!/bin/bash
CUDAID=0
SPECFILE1="psrj0633.pi"
SPECFILE2="pwnj0633.pi"
SPECFILE3="PN_J0633_15asec_grp15.pi"
SPECFILE4="PN_pwn_ex_grp15.pi"
SPECFILE5="M1_J0633_15asec_grp15.pi"
SPECFILE6="M1_pwn_ex_grp15.pi"
SPECFILE7="M2_J0633_15asec_grp15.pi"
SPECFILE8="M2_pwn_ex_grp15.pi"
CHAINFILE=$1
LOGFILE="Log"
NWALK=256
LSTEP=4096
i=$2
NCHAINS=$3
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runGwMcmcOnCuda $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $SPECFILE4 $SPECFILE5 $SPECFILE6 $SPECFILE7 $SPECFILE8 $CHAINFILE $NWALK $LSTEP $i > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
