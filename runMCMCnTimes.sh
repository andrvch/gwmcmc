#!/bin/bash
CUDAID=1
#SPECFILE1="pn-thin-5-ao17_0.fak"
#SPECFILE2="pn-thin-5-ao17_1.fak"
#SPECFILE1="PN_J0633_15asec_grp15_0.fak"
#SPECFILE2="PN_J0633_15asec_grp15_1.fak"
#SPECFILE1="PN_J0633_15asec_bkg.pi"
SPECFILE1="PN_J0633_15asec_grp1.pi"
SPECFILE2="M1_J0633_15asec_grp1.pi"
SPECFILE3="M2_J0633_15asec_grp1.pi"
SPECFILE4="PN_pwn_ex_grp1.pi"
SPECFILE5="M1_pwn_ex_grp1.pi"
SPECFILE6="M2_pwn_ex_grp1.pi"
SPECFILE7="psrj0633.pi"
SPECFILE8="pwnj0633.pi"
SPECFILE9="PN_J0633_15asec_bkg.pi"
SPECFILE10="M1_J0633_bkg.pi"
SPECFILE11="M2_J0633_15asec_bkg.pi"
SPECFILE12="PN_pwn_ex_grp_bkg.pi"
SPECFILE13="M1_pwn_ex_grp_bkg.pi"
SPECFILE14="M2_pwn_ex_grp_bkg.pi"
SPECFILE15="psrj0633_bkg.pi"
SPECFILE16="pwnj0633_bkg.pi"
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
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $SPECFILE4 $SPECFILE5 $SPECFILE6 $SPECFILE7 $SPECFILE8 $SPECFILE9 $SPECFILE10 $SPECFILE11 $SPECFILE12 $SPECFILE13 $SPECFILE14 $SPECFILE15 $SPECFILE16 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
