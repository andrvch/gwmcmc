#!/bin/bash
CUDAID=0
#SPECFILE1="pn-thin-5-ao17_0.fak"
#SPECFILE2="pn-thin-5-ao17_1.fak"
#SPECFILE1="PN_J0633_15asec_grp15.pi"
#SPECFILE2="PN_J0633_15asec_bkg.pi"
#SPECFILE3="PN_pwn_ex_grp15.pi"
#SPECFILE4="PN_pwn_ex_bkg.pi"
SPECFILE1="PN_J0633_15asec_grp15.pi"
SPECFILE2="PN_J0633_15asec_bkg.pi"
SPECFILE3="M1_J0633_15asec_grp15.pi"
SPECFILE4="M1_J0633_bkg.pi"
SPECFILE5="M2_J0633_15asec_grp15.pi"
SPECFILE6="M2_J0633_15asec_bkg.pi"
SPECFILE7="PN_pwn_ex_grp15.pi"
SPECFILE8="PN_pwn_ex_bkg.pi"
SPECFILE9="M1_pwn_ex_grp15.pi"
SPECFILE10="M1_pwn_ex_bkg.pi"
SPECFILE11="M2_pwn_ex_grp15.pi"
SPECFILE12="M2_pwn_ex_bkg.pi"
CHAINFILE=$1
LOGFILE="LogGood"
NWALK=128
LSTEP=512
i=$2
NCHAINS=$3
emin=$4
emax=$5
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $SPECFILE4 $SPECFILE5 $SPECFILE6 $SPECFILE7 $SPECFILE8 $SPECFILE9 $SPECFILE10 $SPECFILE11 $SPECFILE12 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
