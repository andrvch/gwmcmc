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

fitsFls = [sys.argv[1],sys.argv[2],sys.argv[3]] # (files position_input or position_input_11)
regFls = [sys.argv[4],sys.argv[5],sys.argv[6]]

def compareRegAndFits(regFls,fitsFls):
    nim = len(regFls)
    regs = []
    for i in range(nim):
        regs.append(pyregion.open(regFls[i]))
    nst = shape(regs[0])[0]
    id = np.empty([nim,nst])
    for i in range(nim):
        for i in range(nim):
            for j in range(nst):
                id[i,j] = int(re.search(r'\d+',regs[i][j].comment)[0])
    npr = shape(regs[0][0].coord_list)[0]
    ell = np.empty([nim,nst,npr])
    for i in range(nim):
        for j in range(nst):
            for k in range(npr):
                ell[i,int(id[i,j]),k] = regs[i][j].coord_list[k]
    hduls = []
    for i in range(nim):
        hduls.append(fits.open(fitsFls[i]))
    vals = []
    for i in range(nim):
        data = hduls[i][1].data
        n2 = len(data)
        dat = np.empty([n2,4])
        for l in range(n2):
            dat[l,0] = data['X'][l]
            dat[l,1] = data['Y'][l]
            dat[l,2] = data['X_ERR'][l]
            dat[l,3] = data['Y_ERR'][l]
        vals.append(dat)
    coords = np.empty([nim,nst,npr])
    for i in range(nim):
        for j in range(nst):
            res = np.where(np.floor(vals[i][:,0])==math.floor(ell[i,j,0]))
            if len(res[0]) > 0:
                ell[i,j,0] = vals[i][res[0][0],0]
                ell[i,j,1] = vals[i][res[0][0],1]
                ell[i,j,2] = vals[i][res[0][0],2]
                ell[i,j,3] = vals[i][res[0][0],3]
    return ell

ell = compareRegAndFits(regFls,fitsFls)
print(ell[0])

#exit()

hdul = fits.open(fitsFls[0])
#hdul.info()
hdr = hdul[1].header
#print(repr(hdr))

data = hdul[1].data
data.field(0)
print(data[0])

clms = hdul[1].columns
cnms = clms.names

n1 = len(cnms)
n2 = len(data)

print(n1,n2)
print(clms.names)

for i in range(n2):
    print(data['X'][i],data['X_ERR'][i],data['Y_ERR'][i])


exit()

for i in range(n1):
    print(cnms[i],data[0][i])

print(np.floor(pars[0,:,0]))
for i in range(n2):
    res = np.where(np.floor(pars[0,:,0])==math.floor(data['X'][i]))
    if len(res[0]) > 0:
        print(res[0][0],math.floor(data['X'][i]))

f = open(fitsName+".pos", "w")
for i in range(n2):
    f.write("%.15E "%(data['X'][i]))
    f.write("%.15E "%(data['Y'][i]))
    f.write("\n")
f.close()


#print(hdul[1].data)
