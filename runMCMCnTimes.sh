#!/bin/bash
CUDAID=1
SPECFILE1="PN_J0633_15asec_grp1.pi"
SPECFILE2="PN_J0633_15asec_bkg.pi"
SPECFILE3="M1_J0633_15asec_grp1.pi"
SPECFILE4="M1_J0633_bkg.pi"
SPECFILE5="M2_J0633_15asec_grp1.pi"
SPECFILE6="M2_J0633_15asec_bkg.pi"
SPECFILE7="PN_pwn_ex_grp1.pi"
SPECFILE8="PN_pwn_ex_grp_bkg.pi"
SPECFILE9="M1_pwn_ex_grp1.pi"
SPECFILE10="M1_pwn_ex_grp_bkg.pi"
SPECFILE11="M2_pwn_ex_grp1.pi"
SPECFILE12="M2_pwn_ex_grp_bkg.pi"
CHAINFILE=$1
LOGFILE=$2
NWALK=128
LSTEP=$3
i=$4
NCHAINS=$5
emin=$6
emax=$7
let NCHAINS=NCHAINS+i
printf "DeviceID=$CUDAID"
printf "\n"
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $SPECFILE1 $SPECFILE2 $SPECFILE3 $SPECFILE4 $SPECFILE5 $SPECFILE6 $SPECFILE7 $SPECFILE8 $SPECFILE9 $SPECFILE10 $SPECFILE11 $SPECFILE12 $CHAINFILE $NWALK $LSTEP $i $emin $emax > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  ./printSpecStat.py $CHAINFILE
  ./plotSpectraFromFile.py $CHAINFILE
  ./printCrdblFromFile.py $CHAINFILE
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
