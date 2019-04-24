#!/bin/bash
CUDAID=0
#SPECFILE1="pn-thin-5-ao17_0.fak"
#SPECFILE2="pn-thin-5-ao17_1.fak"
#SPECFILE1="PN_J0633_15asec_grp15_0.fak"
#SPECFILE2="PN_J0633_15asec_grp15_1.fak"
#SPECFILE1="PN_J0633_15asec_bkg.pi"
SPECFILE1="PN_J0633_15asec_grp1.pi"
#SPECFILE1="PN_J0633_15asec_grp1.pi"
#SPECFILE2="PN_J0633_15asec_bkg.pi"
SPECFILE2="M1_J0633_15asec_grp1.pi"
#SPECFILE4="M1_J0633_bkg.pi"
SPECFILE3="M2_J0633_15asec_grp1.pi"
#SPECFILE6="M2_J0633_15asec_bkg.pi"
SPECFILE4="PN_pwn_ex_grp1.pi"
#SPECFILE8="PN_pwn_ex_bkg.pi"
SPECFILE5="M1_pwn_ex_grp1.pi"
#SPECFILE10="M1_pwn_ex_bkg.pi"
SPECFILE6="M2_pwn_ex_grp1.pi"
#SPECFILE12="M2_pwn_ex_bkg.pi"
CHAINFILE=$1
LOGFILE=$2
NWALK=128
LSTEP=512
i=$3
NCHAINS=$4
emin=0.4
emax=7.0
let NCHAINS=NCHAINS+i
printf "DeviceID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $SPECFILE4 $SPECFILE5 $SPECFILE6 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
