#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pylab import *
from astropy.io import ascii
from astropy.io import fits
import numpy as np
import pyregion
import re

fitsFl = sys.argv[1] # (files position_input or position_input_11)

hdul = fits.open(fitsFl)
#hdul.info()

hdr = hdul[0].header
#print(repr(hdr))
xrf = hdr['CRVAL1P']
yrf = hdr['CRVAL2P']
xscl = hdr['CDELT1P']
yscl = hdr['CDELT2P']
print(xrf,yrf)
print(xscl,yscl)

data = hdul[0].data

ny = shape(data)[0]
nx = shape(data)[1]

print(nx,ny)

f = open(fitsFl+".psf", "w")
f.write("%.13E\n"%(xrf))
f.write("%.13E\n"%(yrf))
f.write("%.18E\n"%(xscl))
f.write("%.18E\n"%(yscl))
for j in range(ny):
    for i in range(nx):
        f.write("%.15E "%(data[j,i]))
    f.write("\n")
f.close()

exit()


#data.field(0)
#for i in range(len(data)):
#    print(data[i])

#clms = hdul[1].columns
#cnms = clms.names

#n1 = len(cnms)
#n2 = len(data)


#print(clms.names)


for i in range(n2):
    print(data['X'][i],data['X_ERR'][i],data['Y_ERR'][i])

for i in range(n1):
    print(cnms[i],data[0][i])

print(np.floor(pars[0,:,0]))
for i in range(n2):
    res = np.where(np.floor(pars[0,:,0])==math.floor(data['X'][i]))
    if len(res[0]) > 0:
        print(res[0][0],math.floor(data['X'][i]))



#print(hdul[1].data)
