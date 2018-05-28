rmfgen spectrumset=PNsource_spectrum.fits rmfset=PN.rmf

arfgen spectrumset=PNsource_spectrum.fits arfset=PN.arf withrmfset=yes rmfset=PN.rmf \
badpixlocation=PNclean.fits detmaptype=psf

specgroup spectrumset=PNsource_spectrum.fits mincounts=25 oversample=3 rmfset=PN.rmf \
arfset=PN.arf backgndset=PNbackground_spectrum.fits groupedset=PN_spectrum_grp.fits
