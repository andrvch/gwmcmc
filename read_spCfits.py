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
#import pyregion
#import re

fl = "spC.fits"

hdul = fits.open(fl)
print(shape(hdul)[0])

f = open("carb.tab", "w")

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
