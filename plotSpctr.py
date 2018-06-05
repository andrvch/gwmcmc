#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from xspec import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib import rc, font_manager
import pyfits
import matplotlib.patches as mpatches

#Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

SPECNAME = "1:1 PN_J0633_15asec_grp15.pi 2:2 PN_J0633_15asec_bkg.pi"
nspec = 2

ignore_less = "**-0.5"
ignore_more = "8.0-**"

AllData(SPECNAME)
AllData.ignore(ignore_less)
AllData.ignore(ignore_more)
AllData.ignore("bad")

scl = (288000. / 2241600.)

AllModels += "powerlaw + phabs*nsmaxg"
AllModels(1).setPars((0.90, scl*10**-4.61, 0.23, 5.83, 1.4, 10**1.07, 10**(2.57-3.), 1260, 1.))
AllModels(2).setPars((0.90, 10**-4.61, 0.19, 5.96, 1.4, 10**1.07, 10**(2.57-3.), 1260, 0.))

AllData.show()
AllModels.show()
Fit.show()

Fit.statMethod = "chi"
Fit.statTest   = "chi"
print Fit.statistic

Plot.xAxis = "keV"
Plot("data")

spcx   = []
spcy   = []
spcrrx = []
spcrry = []
for i in range(nspec):
    spcx.append(np.array(Plot.x(i+1)))
    spcy.append(np.array(Plot.y(i+1)))
    spcrrx.append(np.array(Plot.xErr(i+1)))
    spcrry.append(np.array(Plot.yErr(i+1)))

mod = []
for i in range(nspec):
    mod.append(np.array(AllModels(i+1).folded(i+1))/spcrrx[i]/2.)

Plot("chi")

chix   = []
chiy   = []
chirrx = []
chirry = []
for i in range(nspec):
    chix.append(np.array(Plot.x(i+1)))
    chiy.append(np.array(Plot.y(i+1)))

E_str = .5  # energy range
E_fin = 8.

gs  = gridspec.GridSpec(8,1)
ax1 = plt.subplot(gs[:5,0])
ax2 = plt.subplot(gs[5:8,0],sharex=ax1)

set_colours = ['g','r','r','r','b','b','k']

#for i in range(nspec):
i = 0
ax1.errorbar(spcx[i],spcy[i],xerr=spcrrx[i],yerr=spcrry[i],color=set_colours[i],fmt=' ',capsize=0)
ax1.step(np.append(spcx[i][0]-spcrrx[i][0],spcx[i]+spcrrx[i]),np.append(mod[i][0],mod[i]),color=set_colours[i])
ax2.step(np.append(chix[i][0]-spcrrx[i][0],chix[i]+spcrrx[i]),np.append(chiy[i][0],chiy[i]),color=set_colours[i])
i = 1
ax1.errorbar(spcx[i],spcy[i],xerr=spcrrx[i],yerr=spcrry[i],color=set_colours[i],fmt=' ',capsize=0)
ax1.step(np.append(spcx[i][0]-spcrrx[i][0],spcx[i]+spcrrx[i]),np.append(mod[i][0],mod[i]),color=set_colours[i])
ax2.step(np.append(chix[i][0]-spcrrx[i][0],chix[i]+spcrrx[i]),np.append(chiy[i][0],chiy[i]),color=set_colours[i])
#red_patch   = mpatches.Patch(color='green', label=r'$\rm src$')
#blu_patch   = mpatches.Patch(color='red',   label=r'$\rm back$')
#green_patch = mpatches.Patch(color='blue',  label=r'$\rm CC$')
#black_patch = mpatches.Patch(color='black', label=r'$\rm ACIS-S$')

#ax1.legend(handles=[red_patch,blu_patch,green_patch,black_patch],loc='upper right',shadow=True)

ax2.plot([E_str,E_fin],[0.0,0.0],'--',color='k')

ax1.set_ylabel(r'$\rm normalized \, counts \, s^{-1} \, keV^{-1} $',fontsize=10)
ax2.set_xlabel(r'$ \rm E  \, [\, \rm keV\,] $',fontsize=10)
ax2.set_ylabel(r'$ \rm sign(data-model)\Delta\chi^{2} $',fontsize=10)

ax1.set_xlim(E_str,E_fin)
ax1.set_ylim(5.E-6,1.)
ax2.set_ylim(-10.,10.)

ax1.set_yscale('log',nonposy='clip')
ax1.set_xscale('log')
ax2.set_xscale('log')

ax2.xaxis.set_major_formatter(LogFormatter(base=10.0,labelOnlyBase=False))
setp(ax1.get_xticklabels(), visible=False)

#ax2.set_xticks(arange(1.,E_fin,1.))
#ax1.set_xticks(arange(1.,E_fin,1.))
#ax2.set_xticks([0.3, 0.5, 0.7, 1, 2, 3, 5])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax2.set_yticks(arange(-.02,.02+0.001,0.02))

plt.savefig('spctr.eps')
#plt.show()
