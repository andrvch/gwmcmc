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

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"
Fit.statMethod = "chi"
Fit.statTest   = "chi"

SPECNAME = "1:1 PN_J0633_15asec.pi 2:2 PN_J0633_15asec_bkg.pi"
nspec = 2

ignore_less = "**-0.5"
ignore_more = "7.0-**"

AllData(SPECNAME)
AllData.ignore(ignore_less)
AllData.ignore(ignore_more)
AllData.ignore("bad")

gr = math.sqrt(1-2.952*1.4/10**1.08)
scl = (288000. / 2241600.)
bckPhIndx = 0.91
bckNrm = -4.99
nh = 0.31
Teff = 5.807
Mns = 1.4
logR = 0.9232
magfld = 1e12
logD = 2.593
psrPhIndx = 1.25
psrNrm = -5.186

AllModels += "(nsa+powerlaw)*phabs + powerlaw"
AllModels(1).setPars((Teff, Mns, 10**logR, magfld, 10**(-2*logD), psrPhIndx, 10**psrNrm, nh, bckPhIndx, scl*10**bckNrm))
AllModels(2).setPars((Teff, Mns, 10**logR, magfld, 0., psrPhIndx, 0., nh, bckPhIndx, 10**bckNrm))

Fit.show()
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
E_fin = 7.

gs  = gridspec.GridSpec(8,1)
ax1 = plt.subplot(gs[:5,0])
ax2 = plt.subplot(gs[5:8,0],sharex=ax1)

set_colours = ['g','gray']

#for i in range(nspec):
i = 0
ax1.errorbar(spcx[i],spcy[i],xerr=spcrrx[i],yerr=spcrry[i],color=set_colours[i],fmt=' ',capsize=0)
ax1.step(np.append(spcx[i][0]-spcrrx[i][0],spcx[i]+spcrrx[i]),np.append(mod[i][0],mod[i]),color=set_colours[i])
ax2.step(np.append(chix[i][0]-spcrrx[i][0],chix[i]+spcrrx[i]),np.append(chiy[i][0],chiy[i]),color=set_colours[i])
i = 1
ax1.errorbar(spcx[i],scl*spcy[i],xerr=spcrrx[i],yerr=scl*spcrry[i],color=set_colours[i],fmt=' ',capsize=0)
ax1.step(np.append(spcx[i][0]-spcrrx[i][0],spcx[i]+spcrrx[i]),np.append(scl*mod[i][0],scl*mod[i]),color=set_colours[i])
ax2.step(np.append(chix[i][0]-spcrrx[i][0],chix[i]+spcrrx[i]),np.append(scl*chiy[i][0],scl*chiy[i]),color=set_colours[i])

ax2.plot([E_str,E_fin],[0.0,0.0],'--',color='k')

#ax1.set_ylabel(r'$\rm normalized \, counts \, s^{-1} \, keV^{-1} $',fontsize=10)
#ax2.set_xlabel(r'$ \rm E  \, [\, \rm keV\,] $',fontsize=10)
#ax2.set_ylabel(r'$ \rm sign(data-model)\Delta\chi^{2} $',fontsize=10)

ax1.set_xlim(E_str,E_fin)
ax1.set_ylim(5.E-6,2.E-1)
#ax2.set_ylim(-10.,10.)

ax1.set_yscale('log',nonposy='clip')
ax1.set_xscale('log')
ax2.set_xscale('log')

ax2.xaxis.set_major_formatter(LogFormatter(base=10.0,labelOnlyBase=False))
setp(ax1.get_xticklabels(), visible=False)

ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.savefig('spctr.eps')
#plt.show()
