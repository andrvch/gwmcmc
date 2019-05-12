#!/bin/bash
CUDAID=1
#SPECFILE1="pn-thin-5-ao17_0.fak"
#SPECFILE2="pn-thin-5-ao17_1.fak"
SPECFILE1="PN_J0633_15asec_grp1.pi"
BACKFILE1="PN_J0633_15asec_bkg.pi"
SPECFILE2="M1_J0633_15asec_grp1.pi"
BACKFILE2="M1_J0633_bkg.pi"
SPECFILE3="M2_J0633_15asec_grp1.pi"
BACKFILE3="M2_J0633_15asec_bkg.pi"
SPECFILE4="PN_pwn_ex_grp1.pi"
BACKFILE4="PN_pwn_ex_grp_bkg.pi"
SPECFILE5="M1_pwn_ex_grp1.pi"
BACKFILE5="M1_pwn_ex_grp_bkg.pi"
SPECFILE6="M2_pwn_ex_grp1.pi"
BACKFILE6="M2_pwn_ex_grp_bkg.pi"
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
  ./runSFH $CUDAID $SPECFILE1 $BACKFILE1 $SPECFILE2 $BACKFILE2 $SPECFILE3 $BACKFILE3 $SPECFILE4 $BACKFILE4 $SPECFILE5 $BACKFILE5 $SPECFILE6 $BACKFILE6 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
