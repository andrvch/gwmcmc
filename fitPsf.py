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
#from cudakde import *

fl = sys.argv[1]

hdul = fits.open(fl)
hdr = hdul[0].header
#print(repr(hdr))

xrf = hdr['CRVAL1P']
yrf = hdr['CRVAL2P']
xscl = hdr['CDELT1P']
yscl = hdr['CDELT2P']

print(xrf+20.5,yrf+20.5)
print(xscl,yscl)

data = hdul[0].data

ny = shape(data)[0]
nx = shape(data)[1]

print(nx,ny)

#exit()
f = open(fl+".newpsf", "w")

#f.write("%.13E\n"%(xrf))
#f.write("%.13E\n"%(yrf))
#f.write("%.18E\n"%(xscl))
#f.write("%.18E\n"%(yscl))
tot = data.sum()
print("Total:")
print(tot)

for j in range(ny):
    for i in range(nx):
        f.write("%.15E "%(data[j,i]))
    f.write("\n")
f.close()

xii,yii = np.mgrid[1.:nx:nx*1j,1.:ny:ny*1j]
#print(xii)
#for i in range(nx):
#    print(data[i,:])
#print(data)

def comp_lev(zi,quont):
    zisort = np.sort(zi)[::-1]
    zisum = np.sum(zi)
    zicumsum = np.cumsum(zisort)
    zicumsumnorm = zicumsum/zisum
    #print(zicumsumnorm)
    #print(np.where(zicumsumnorm>90)[0])
    zinorm = zi/zisum
    zisortnorm = zisort/zisum
    levels = [zisortnorm[np.where(zicumsumnorm>qu)][0] for qu in quont]
    return levels,zinorm,zisum

"""
quont = [0.90,0.68,0.40]
lev,zin,zisum = comp_lev(data.flatten(),quont)

print(zisum)
print(lev)

for i in range(len(quont)):
    print(lev[i]*zisum)

plt.contour(xii,yii,zin.reshape(xii.shape), lev, colors='blue', linewidth=.5)

plt.savefig(sys.argv[1]+"cntr"+".png")
"""

"""
def gauss(x,y):
    return np.exp(-0.5*(x**2+y**2))
zii = gauss(xii,yii)
levi,ziin = comp_lev(zii.flatten(),quont)
"""
