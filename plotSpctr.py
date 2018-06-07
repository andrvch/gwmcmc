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

SPECNAME = "1:1 PN_J0633_15asec_grp15.pi 2:2 PN_J0633_15asec_bkg.pi 3:3 M1_J0633_15asec_grp15.pi 4:4 M1_J0633_bkg.pi 5:5 M2_J0633_15asec_grp15.pi 6:6 M2_J0633_15asec_bkg.pi 7:7 PN_pwn_ex_grp15.pi 8:8 PN_pwn_ex_bkg.pi 9:9 M1_pwn_ex_grp15.pi 10:10 M1_pwn_ex_bkg.pi 11:11 M2_pwn_ex_grp15.pi 12:12 M2_pwn_ex_bkg.pi"

erange = [0.5, 5.0]
nspec = 12

AllData(SPECNAME)
AllData.ignore("**-%2.1f %2.1f-**"%(erange[0],erange[1]))
AllData.ignore("bad")

scl = [288000. / 2241600., 271732. / 2207424., 286400. / 2241600., 2595200. / 2241600., 2574576. / 2207424., 2465192. / 2241600.]

bckIndx = [0.92, 0.87, 1.01, 0.88, 1.12, 1.13 ]
bckNrm = [-5.00, -5.14, -5.09, -5.00, -5.08, -5.05]

gr = 10**-0.08
Mns = 1.4
Rns = 2.952 * Mns / ( 1 - gr**2 )

nh = 0.21
Teff = 5.77
logR = math.log10(Rns)
magfld = 1.e12
logD = 2.88
psrIndx = 1.33
psrNrm = -5.19
pwnIndx = 1.84
pwnNrm = -4.57

AllModels += "(nsa+powerlaw)*phabs+powerlaw"
for i in range(int(nspec/2./2.)):
    AllModels(2*i+1).setPars((Teff, Mns, 10**logR, magfld, 10**(-2.*logD), psrIndx, 10**psrNrm, nh, bckIndx[i], scl[i]*10**bckNrm[i]))
    AllModels(2*i+2).setPars((Teff, Mns, 10**logR, magfld, 0., psrIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
    AllModels(2*i+1+int(nspec/2.)).setPars((Teff, Mns, 10**logR, magfld, 0., pwnIndx, 10**pwnNrm, nh, bckIndx[i+int(nspec/2./2.)], scl[i+int(nspec/2./2.)]*10**bckNrm[i+int(nspec/2./2.)]))
    AllModels(2*i+2+int(nspec/2.)).setPars((Teff, Mns, 10**logR, magfld, 0., pwnIndx, 0., nh, bckIndx[i+int(nspec/2./2.)], 10**bckNrm[i+int(nspec/2./2.)]))

Fit.show()
print Fit.statistic

Plot.xAxis = "keV"
Plot("data")

spcx = []
spcy = []
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

Plot("resid")

chix = []
chiy = []
chirrx = []
chirry = []

for i in range(nspec):
    chix.append(np.array(Plot.x(i+1)))
    chiy.append(np.array(Plot.y(i+1)))
    chirrx.append(np.array(Plot.xErr(i+1)))
    chirry.append(np.array(Plot.yErr(i+1)))

fig, ax = plt.subplots(nrows=4)
gs = gridspec.GridSpec(14,1)

ax[0] = plt.subplot(gs[:4,0])
ax[1] = plt.subplot(gs[4:8,0])
ax[2] = plt.subplot(gs[8:12,0])
ax[3] = plt.subplot(gs[12:14,0])

setcolours = ['b','g','r','c','m','y']

for i in range(int(nspec/2./2.)):
    ax[0].errorbar(spcx[2*i],spcy[2*i],xerr=spcrrx[2*i],yerr=spcrry[2*i],color=setcolours[i],fmt=' ',capsize=0)
    ax[0].step(np.append(spcx[2*i][0]-spcrrx[2*i][0],spcx[2*i]+spcrrx[2*i]),np.append(mod[2*i][0],mod[2*i]),color=setcolours[i])
    ax[1].errorbar(spcx[2*i+int(nspec/2.)],spcy[2*i+int(nspec/2.)],xerr=spcrrx[2*i+int(nspec/2.)],yerr=spcrry[2*i+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
    ax[1].step(np.append(spcx[2*i+int(nspec/2.)][0]-spcrrx[2*i+int(nspec/2.)][0],spcx[2*i+int(nspec/2.)]+spcrrx[2*i+int(nspec/2.)]),np.append(mod[2*i+int(nspec/2.)][0],mod[2*i+int(nspec/2.)]),color=setcolours[i+int(nspec/2./2.)])
    ax[2].errorbar(spcx[2*i+1],scl[i]*spcy[2*i+1],xerr=spcrrx[2*i+1],yerr=scl[i]*spcrry[2*i+1],color=setcolours[i],fmt=' ',capsize=0)
    ax[2].errorbar(spcx[2*i+1+int(nspec/2.)],scl[i+int(nspec/2./2.)]*spcy[2*i+1+int(nspec/2.)],xerr=spcrrx[2*i+1+int(nspec/2.)],yerr=scl[i+int(nspec/2./2.)]*spcrry[2*i+1+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
    ax[2].step(np.append(spcx[2*i+1][0]-spcrrx[2*i+1][0],spcx[2*i+1]+spcrrx[2*i+1]),np.append(scl[i]*mod[2*i+1][0],scl[i]*mod[2*i+1]),color=setcolours[i])
    ax[2].step(np.append(spcx[2*i+1+int(nspec/2.)][0]-spcrrx[2*i+1+int(nspec/2.)][0],spcx[2*i+1+int(nspec/2.)]+spcrrx[2*i+1+int(nspec/2.)]),np.append(scl[i+int(nspec/2./2.)]*mod[2*i+1+int(nspec/2.)][0],scl[i+int(nspec/2./2.)]*mod[2*i+1+int(nspec/2.)]),color=setcolours[i+int(nspec/2./2.)])
    ax[3].errorbar(spcx[2*i],chiy[2*i],xerr=spcrrx[2*i],yerr=chirry[2*i],color=setcolours[i],fmt=' ',capsize=0)
    ax[3].errorbar(spcx[2*i+int(nspec/2./2.)],chiy[2*i+int(nspec/2./2.)],xerr=spcrrx[2*i+int(nspec/2./2.)],yerr=chirry[2*i+int(nspec/2./2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
    ax[3].errorbar(spcx[2*i+1],scl[i]*chiy[2*i+1],xerr=spcrrx[2*i+1],yerr=scl[i]*chirry[2*i+1],color=setcolours[i],fmt=' ',capsize=0)
    ax[3].errorbar(spcx[2*i+1+int(nspec/2.)],scl[i+int(nspec/2./2.)]*chiy[2*i+1+int(nspec/2.)],xerr=spcrrx[2*i+1+int(nspec/2.)],yerr=scl[i+int(nspec/2./2.)]*chirry[2*i+1+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)

ax[3].plot(erange,[0.0,0.0],'--',color='k')

for i in range(4):
    ax[i].set_xlim(erange[0],erange[1])
    ax[i].set_xscale('log')

for i in range(3):
    ax[i].set_yscale('log',nonposy='clip')
    ax[i].xaxis.set_major_formatter(LogFormatter(base=10.0,labelOnlyBase=False))
    ax[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

setp([a.get_xticklabels() for a in ax[:4-1]], visible=False)

#ax[i].set_ylabel(r'$\rm normalized \, counts \, s^{-1} \, keV^{-1} $',fontsize=10)
#ax[3].set_xlabel(r'$ \rm E  \, [\, \rm keV\,] $',fontsize=10)
#ax[3].set_ylabel(r'$ \rm sign(data-model)\Delta\chi^{2} $',fontsize=10)

plt.savefig('psrpwnspctr.eps')
#plt.show()
