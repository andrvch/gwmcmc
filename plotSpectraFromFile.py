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
spcs = readspectra(nspec,sys.argv[1]+"spec"+".spec")
erange = [0.3,10.0]

fig, ax = plt.subplots(nrows=2)
gs = gridspec.GridSpec(3,1)
ax[0] = plt.subplot(gs[:2,0])
ax[1] = plt.subplot(gs[2:3,0])

setcolours = [ 'g', 'b', 'r' ]
bkgcolours = [ 'gray', 'gray', 'gray' ]

for i in range(nspec/2):
    nbins = shape(spcs[i])[1]
    xxen = 0.5*(spcs[i][0]+spcs[i][1])
    xxenerr = 0.5*(spcs[i][1]-spcs[i][0])
    ax[0].errorbar(xxen,spcs[i][2]-spcs[i][3],xerr=xxenerr,yerr=np.sqrt(spcs[i][2]+spcs[i][3]),color=setcolours[i],fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i][3],xerr=xxenerr,yerr=np.sqrt(spcs[i][3]),color=bkgcolours[i],alpha=0.25,fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i][8],xerr=xxenerr,yerr=np.sqrt(spcs[i][8]),color=setcolours[i],fmt=' ',capsize=0)
    ax[0].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i][4][0],spcs[i][4]),color=setcolours[i])
    ax[0].errorbar(xxen,spcs[i][5],color=setcolours[i],fmt='--')
    ax[0].errorbar(xxen,spcs[i][6],color=setcolours[i],fmt='-.')
    #ax[0].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i][5][0],spcs[i][5]),alpha=0.25,color=setcolours[i])
    ax[1].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i][9][0],spcs[i][9]),color=setcolours[i])

subs = [1., 2., 3., 4.0, 5.0, 7.]

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')

for i in range(2):
    ax[i].set_xlim(erange[0],erange[1])
    #ax[i].set_xscale('log')
    #ax[i].xaxis.set_major_formatter(CustomTicker())
    ax[i].xaxis.set_minor_locator(ticker.LogLocator(subs=subs)) #set the ticks position
    ax[i].xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
    ax[i].tick_params(axis='both',which='major',labelsize=10)
    ax[i].tick_params(axis='both',which='minor',labelsize=10)

ax[1].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  #add the custom ticks

plt.setp(ax[1].get_xticklabels(minor=True),visible=True)

plt.setp([a.get_xticklabels() for a in ax[:1]], visible=False)
plt.setp([a.get_xticklabels(minor=True) for a in ax[:1]], visible=False)

ax[0].set_ylim(0.1,150.0)
ax[1].set_ylim(-5.,5.)
ax[1].set_yticks(np.arange(-5., 5.1, step=2.5))

ax[1].set_xlabel(r'$ \rm Photon \, Energy  \, [\, \rm keV\,] $',fontsize=10)
ax[0].set_ylabel(r'$ \rm Counts $',fontsize=10)
ax[1].set_ylabel(r'$ \chi^{2} $',fontsize=10)

l = ax[0].legend(['pn','MOS1','MOS2'],fontsize=9,loc=1)
l.set_zorder(5)

plt.savefig(sys.argv[1]+"spectraPSR"+".jpg")
plt.savefig(sys.argv[1]+"spectraPSR"+".eps")
#plt.show()

fig, ax = plt.subplots(nrows=2)
gs = gridspec.GridSpec(3,1)
ax[0] = plt.subplot(gs[:2,0])
ax[1] = plt.subplot(gs[2:3,0])

setcolours = [ 'g', 'b', 'r' ]

for i in range(nspec/2):
    nbins = shape(spcs[i+nspec/2])[1]
    xxen = 0.5*(spcs[i+nspec/2][0]+spcs[i+nspec/2][1])
    xxenerr = 0.5*(spcs[i+nspec/2][1]-spcs[i+nspec/2][0])
    ax[0].errorbar(xxen,spcs[i+nspec/2][2]-spcs[i+nspec/2][3],xerr=xxenerr,yerr=np.sqrt(spcs[i+nspec/2][2]+spcs[i+nspec/2][3]),color=setcolours[i],fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i+nspec/2][3],xerr=xxenerr,yerr=np.sqrt(spcs[i+nspec/2][3]),alpha=0.25,color=bkgcolours[i],fmt=' ',capsize=0)
    #ax[0].errorbar(xxen,spcs[i+nspec/2][8],xerr=xxenerr,yerr=np.sqrt(spcs[i+nspec/2][8]),color=setcolours[i],fmt=' ',capsize=0)
    ax[0].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i+nspec/2][4][0],spcs[i+nspec/2][4]),color=setcolours[i])
    #ax[0].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i+nspec/2][5][0],spcs[i+nspec/2][5]),alpha=0.25,color=setcolours[i])
    ax[1].step(np.append(xxen[0]-xxenerr[0],xxen+xxenerr),np.append(spcs[i+nspec/2][9][0],spcs[i+nspec/2][9]),color=setcolours[i])

subs = [1., 2., 3., 4., 5., 7.]

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')

for i in range(2):
    ax[i].set_xlim(erange[0],erange[1])
    #ax[i].set_xscale('log')
    #ax[i].xaxis.set_major_formatter(CustomTicker())
    ax[i].xaxis.set_minor_locator(ticker.LogLocator(subs=subs)) #set the ticks position
    ax[i].xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
    ax[i].tick_params(axis='both', which='major', labelsize=10)
    ax[i].tick_params(axis='both', which='minor', labelsize=10)

ax[1].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  #add the custom ticks

plt.setp(ax[1].get_xticklabels(minor=True), visible=True)

plt.setp([a.get_xticklabels() for a in ax[:1]], visible=False)
plt.setp([a.get_xticklabels(minor=True) for a in ax[:1]], visible=False)

ax[0].set_ylim(0.1,150.0)
ax[1].set_ylim(-5.,5.)
ax[1].set_yticks(np.arange(-5., 5.1, step=2.5))

#ax[i].set_ylabel(r'$\rm normalized \, counts \, s^{-1} \, keV^{-1} $',fontsize=10)
ax[1].set_xlabel(r'$ \rm Photon \, energy  \, [\, \rm keV\,] $',fontsize=10)
ax[0].set_ylabel(r'$ \rm Counts $',fontsize=10)
ax[1].set_ylabel(r'$ \chi^{2} $',fontsize=10)

l = ax[0].legend(['pn','MOS1','MOS2'],fontsize=9,loc=1)
l.set_zorder(5)

plt.savefig(sys.argv[1]+"specPWN"+".jpg")
plt.savefig(sys.argv[1]+"specPWN"+".eps")
#plt.show()
