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

fig, ax = plt.subplots(ncols=1, nrows=2)
setcolours = [ 'g', 'b', 'r' ]
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')

for i in range(nspec/2):
    nbins = shape(spcs[i])[1]
    xxen = 0.5*(spcs[i][0]+spcs[i][1])
    xxenerr = 0.5*(spcs[i][1]-spcs[i][0])
    ax[0].errorbar(xxen,spcs[i][2]-spcs[i][3],xerr=xxenerr,yerr=np.sqrt(spcs[i][2]+spcs[i][3]),color=setcolours[i],fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i][3],xerr=xxenerr,yerr=np.sqrt(spcs[i][3]),color=setcolours[i],fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i][4],xerr=xxenerr,yerr=np.sqrt(spcs[i][4]),color=setcolours[i],fmt=' ',capsize=0)
    ax[0].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i][5][0],spcs[i][5]),color=setcolours[i])
    ax[1].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i][7][0],spcs[i][7]),color=setcolours[i])

plt.savefig("spectraPSR"+".jpg")
#plt.show()

fig, ax = plt.subplots(ncols=1, nrows=2)
setcolours = [ 'g', 'b', 'r' ]
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')

for i in range(nspec/2):
    nbins = shape(spcs[i+nspec/2])[1]
    xxen = 0.5*(spcs[i+nspec/2][0]+spcs[i+nspec/2][1])
    xxenerr = 0.5*(spcs[i+nspec/2][1]-spcs[i+nspec/2][0])
    ax[0].errorbar(xxen,spcs[i+nspec/2][2]-spcs[i+nspec/2][3],xerr=xxenerr,yerr=np.sqrt(spcs[i+nspec/2][2]+spcs[i+nspec/2][3]),color=setcolours[i],fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i+nspec/2][3],xerr=xxenerr,yerr=np.sqrt(spcs[i+nspec/2][3]),color=setcolours[i+nspec/2-nspec/2],fmt=' ',capsi+nspec/2ze=0)
    ax[0].errorbar(xxen,spcs[i+nspec/2][5],xerr=xxenerr,yerr=np.sqrt(spcs[i+nspec/2][5]),color=setcolours[i],fmt=' ',capsize=0)
    ax[1].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i+nspec/2][7][0],spcs[i+nspec/2][7]),color=setcolours[i])

plt.savefig("spectraPWN"+".jpg")
#plt.show()
