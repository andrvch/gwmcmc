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
Fit.statTest = "chi"

nspec = 6
SPECNAME = "1:1 PN_J0633_15asec.pi 2:2 PN_J0633_15asec_bkg.pi 3:3 M1_J0633_15asec.pi 4:4 M1_J0633_bkg.pi 5:5 M2_J0633_15asec_grp15.pi 6:6 M2_J0633_15asec_bkg.pi"
AllData(SPECNAME)
AllData.ignore("**-0.5 7.0-**")
AllData.ignore("bad")

scl = [288000. / 2241600., 271732. / 2207424., 286400. / 2241600.]
bckPhIndx = [0.89, 1.13, 1.19]
bckNrm = [-5.00, -5.09, -5.05]

nh = 0.23
Teff = 5.77
Mns = 1.4
logR = 1.113
magfld = 1.e12
logD = 2.89
psrPhIndx = 1.6
psrNrm = -5.08

#AllModels += "(nsmaxg+powerlaw)*phabs + powerlaw"
#AllModels(1).setPars((Teff, Mns, 10**logR, 10**(logD-3.), magfld, 1., psrPhIndx, 10**psrNrm, nh, bckPhIndx1, scl1*10**bckNrm1))
#AllModels(2).setPars((Teff, Mns, 10**logR, 10**(logD-3.), magfld, 0., psrPhIndx, 0., nh, bckPhIndx1, 10**bckNrm1))
#AllModels(3).setPars((Teff, Mns, 10**logR, 10**(logD-3.), magfld, 1., psrPhIndx, 10**psrNrm, nh, bckPhIndx2, scl2*10**bckNrm2))
#AllModels(4).setPars((Teff, Mns, 10**logR, 10**(logD-3.), magfld, 0., psrPhIndx, 0., nh, bckPhIndx2, 10**bckNrm2))

AllModels += "(nsa+powerlaw)*phabs + powerlaw"
for i in range(int(nspec/2.)):
    AllModels(2*i+1).setPars((Teff, Mns, 10**logR, magfld, 10**(-2.*logD), psrPhIndx, 10**psrNrm, nh, bckPhIndx[i], scl[i]*10**bckNrm[i]))
    AllModels(2*i+2).setPars((Teff, Mns, 10**logR, magfld, 0., psrPhIndx, 0., nh, bckPhIndx[i], 10**bckNrm[i]))

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
E_fin = 3.

gs  = gridspec.GridSpec(8,1)
ax1 = plt.subplot(gs[:5,0])
ax2 = plt.subplot(gs[5:8,0],sharex=ax1)

set_colours = ['g','gray','b','gray','y','gray']

for i in range(int(nspec/2.)):
    ax1.errorbar(spcx[2*i],spcy[2*i],xerr=spcrrx[2*i],yerr=spcrry[2*i],color=set_colours[2*i],fmt=' ',capsize=0)
    ax1.step(np.append(spcx[2*i][0]-spcrrx[2*i][0],spcx[2*i]+spcrrx[2*i]),np.append(mod[2*i][0],mod[2*i]),color=set_colours[2*i])
    ax2.step(np.append(chix[2*i][0]-spcrrx[2*i][0],chix[2*i]+spcrrx[2*i]),np.append(chiy[2*i][0],chiy[2*i]),color=set_colours[2*i])
    ax1.errorbar(spcx[2*i+1],scl[i]*spcy[2*i+1],xerr=spcrrx[2*i+1],yerr=scl[i]*spcrry[2*i+1],color=set_colours[2*i+1],fmt=' ',capsize=0)
    ax1.step(np.append(spcx[2*i+1][0]-spcrrx[2*i+1][0],spcx[2*i+1]+spcrrx[2*i+1]),np.append(scl[i]*mod[2*i+1][0],scl[i]*mod[2*i+1]),color=set_colours[2*i+1])
    ax2.step(np.append(chix[2*i+1][0]-spcrrx[2*i+1][0],chix[2*i+1]+spcrrx[2*i+1]),np.append(scl[i]*chiy[2*i+1][0],scl[i]*chiy[2*i+1]),color=set_colours[2*i+1])

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
