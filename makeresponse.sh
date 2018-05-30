#!/bin/bash
EVTFILE="PNclean.fits"
SPIFILE="spectrumPN_pwn_ex_bkg.fits"
RMFFILE="spectrumPN_pwn_ex_bkg.rmf"
ARFFILE="spectrumPN_pwn_ex_bkg.arf"
SPOFILE="PN_pwn_ex_bkg.pi"
rmfgen spectrumset=$SPIFILE rmfset=$RMFFILE

arfgen spectrumset=$SPIFILE arfset=$ARFFILE withrmfset=yes rmfset=$RMFFILE \
badpixlocation=$EVTFILE detmaptype=psf

specgroup spectrumset=$SPIFILE mincounts=1 oversample=3 rmfset=$RMFFILE \
arfset=$ARFFILE groupedset=$SPOFILE
