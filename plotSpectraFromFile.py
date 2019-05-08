#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import ticker
from cudakde import *

nspec = 6
spcs = readspectra(nspec,sys.argv[1])
nnn = int(len(spcs)/6)
print nnn

fig, ax = plt.subplots(ncols=1, nrows=2)
setcolours = [ 'g', 'b', 'r' ]

nbins = shape(spcs[0])[1]
xxbins = np.linspace(0,nbins,nbins)
xxen = 0.5*(spcs[0][0]+spcs[0][1])
xxenerr = 0.5*(spcs[0][1]-spcs[0][0])

print xxen
print xxenerr

ax[0].errorbar(xxen,spcs[0][2],xerr=xxenerr,yerr=np.sqrt(spcs[0][2]),color=setcolours[0],fmt=' ',capsize=0)
ax[0].errorbar(xxen,spcs[0][3],xerr=xxenerr,yerr=np.sqrt(spcs[0][3]),color=setcolours[0],fmt=' ',capsize=0)
ax[0].errorbar(xxen,spcs[0][4],xerr=xxenerr,yerr=np.sqrt(spcs[0][3]),color=setcolours[0],fmt=' ',capsize=0)
ax[1].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[0][5][0],spcs[0][5]),color='g')

ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[0].set_yscale('log')

plt.savefig(sys.argv[1]+".spectra"+".jpg")
#plt.show()
