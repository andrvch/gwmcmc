#!/bin/bash
EVTFILE="PNclean.fits"
SPIFILE="spectrumM2_J0633_15asec_bkg.fits"
RMFFILE="spectrumM2_J0633_15asec_bkg.rmf"
ARFFILE="spectrumM2_J0633_15asec_bkg.arf"
SPOFILE="M2_J0633_15asec_bkg.pi"
rmfgen spectrumset=$SPIFILE rmfset=$RMFFILE

arfgen spectrumset=$SPIFILE arfset=$ARFFILE withrmfset=yes rmfset=$RMFFILE \
badpixlocation=$EVTFILE detmaptype=psf

specgroup spectrumset=$SPIFILE mincounts=1 oversample=3 rmfset=$RMFFILE \
arfset=$ARFFILE groupedset=$SPOFILE
