#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import os, sys
import math
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from astropy.io import ascii
from astropy.io import fits

fl = "spC.fits"

hdul = fits.open(fl)
print(shape(hdul)[0])

f = open("carb.tab", "w")

nme = shape(hdul[1].data)[0]
nmg = shape(hdul[2].data)[0]
nmt = shape(hdul[3].data)[0]
print(nme,nmg,nmt)

en = cp.empty(nme,dtype=np.float32)
gr = cp.empty(nmg,dtype=np.float32)
te = cp.empty(nmt,dtype=np.float32)

fx = cp.empty((nmg,nmt,nme),dtype=np.float32)

for i in range(nme):
    en[i] = float(hdul[1].data[i][0])
print(en)

for i in range(nmg):
    gr[i] = float(hdul[2].data[i][0])
print(gr)

for i in range(nmt):
    te[i] = float(hdul[3].data[i][0])
print(te)

for i in range(nmg):
    for j in range(nmt):
        for k in range(nme):
            fx[i,j,k] = float(hdul[4+i].data[i][0][k])
print(fx)

cp.save('catm_energy.npy',en)
cp.save('catm_gravsh.npy',gr)
cp.save('catm_tempef.npy',te)
cp.save('catm_fluxes.npy',fx)

exit()
for i in range(shape(hdul)[0]):
    print("%i:"%i)
    hdr = hdul[i].header
    #print(repr(hdr))
    data = hdul[i].data
    print(shape(data))
    #if i < 5:
    #print(data)
    if i == 1:
        for j in range(shape(data)[0]):
            #for k in range(2):
            f.write(" %.15E "%(data[j][0]))
            f.write("\n")
        #f.write("\n")
    if i == 2 or i == 3:
        for j in range(shape(data)[0]):
            f.write(" %.15E "%(data[j][0]))
        #f.write("\n")
        #f.write("\n")
    if 3 < i:
        #print(data)
        print(type(data))
        for k in range(61):
            for j in range(1000):
                f.write(" %.15E "%(data[k][0][j]))
            f.write("\n")
        #f.write("\n")

f.close()
