#!/bin/bash
EVTFILE="PNclean.fits"
SPIFILE="spectrumPN_J0633_15asec_bkg.fits"
RMFFILE="spectrumPN_J0633_15asec_bkg.rmf"
ARFFILE="spectrumPN_J0633_c_15asec_bkg.arf"
SPOFILE="PN_J0633_15asec_bkg.pi"
rmfgen spectrumset=$SPIFILE rmfset=$RMFFILE

arfgen spectrumset=$SPIFILE arfset=$ARFFILE withrmfset=yes rmfset=$RMFFILE \
badpixlocation=$EVTFILE detmaptype=psf

specgroup spectrumset=$SPIFILE mincounts=1 oversample=3 rmfset=$RMFFILE \
arfset=$ARFFILE groupedset=$SPOFILE
