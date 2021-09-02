#!/bin/bash
CUDAID=0
CHAINFILE="psffitTest001_j2030+4415_"
LOGFILE="LogPsfFitTest001_j2030+4415"
NWALK=128
LSTEP=1024
IMDIM=40
i=$1
NCHAINS=5
PSFFL1="psf_14827_src1_1pix.fits.newpsf"
DATAFL1="img_14827_src1.fits.newpsf"
PSFFL2="psf_14827_src2_1pix.fits.newpsf"
DATAFL2="img_14827_src2.fits.newpsf"
PSFFL3="psf_14827_src3_1pix.fits.newpsf"
DATAFL3="img_14827_src3.fits.newpsf"
PSFFL4="psf_14827_psr_1pix.fits.newpsf"
DATAFL4="img_14827_psr.fits.newpsf"
PSFFL5="psf_20298_src1_1pix.fits.newpsf"
DATAFL5="img_20298_src1.fits.newpsf"
PSFFL6="psf_20298_src2_1pix.fits.newpsf"
DATAFL6="img_20298_src2.fits.newpsf"
PSFFL7="psf_20298_src3_1pix.fits.newpsf"
DATAFL7="img_20298_src3.fits.newpsf"
PSFFL8="psf_20298_psr_1pix.fits.newpsf"
DATAFL8="img_20298_psr.fits.newpsf"
let NCHAINS=NCHAINS+i
printf "Start>"
while [ $i -lt $NCHAINS ]; do
  ./runSFH $CUDAID $CHAINFILE $NWALK $LSTEP $i $IMDIM $PSFFL1 $DATAFL1 $PSFFL2 $DATAFL2 $PSFFL3 $DATAFL3 $PSFFL4 $DATAFL4 $PSFFL5 $DATAFL5 $PSFFL6 $DATAFL6 $PSFFL7 $DATAFL7 $PSFFL8 $DATAFL8 > $LOGFILE
  ./plotChain.py $CHAINFILE $i $NWALK
  let i=i+1
  let TOTAL=i*LSTEP
  printf "$TOTAL"
  printf ">"
done
printf "Stop"
printf "\n"
