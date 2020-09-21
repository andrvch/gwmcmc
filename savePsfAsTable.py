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

hdul = fits.open(fitsFls[0])
#hdul.info()
hdr = hdul[0].header
#print(repr(hdr))

data = hdul[0].data
#data.field(0)
for i in range(len(data)):
    print(data[i])

#clms = hdul[1].columns
#cnms = clms.names

#n1 = len(cnms)
#n2 = len(data)

#print(n1,n2)
#print(clms.names)

exit()


for i in range(n2):
    print(data['X'][i],data['X_ERR'][i],data['Y_ERR'][i])



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
