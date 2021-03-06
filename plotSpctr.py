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
from xspec import *
from cudakde import *

psr = int(sys.argv[1])

#Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"
Fit.statMethod = "chi"
Fit.statTest = "chi"

erange = [0.2, 10.0]
erange2 = [0.4, 7.0]

SPECNAME = "1:1 PN_J0633_15asec_grp1.pi 2:2 PN_J0633_15asec_bkg.pi 3:3 M1_J0633_15asec_grp1.pi 4:4 M1_J0633_bkg.pi 5:5 M2_J0633_15asec_grp1.pi 6:6 M2_J0633_15asec_bkg.pi 7:7 PN_pwn_ex_grp1.pi 8:8 PN_pwn_ex_bkg.pi 9:9 M1_pwn_ex_grp1.pi 10:10 M1_pwn_ex_bkg.pi 11:11 M2_pwn_ex_grp1.pi 12:12 M2_pwn_ex_bkg.pi"
#SPECNAME = "1:1 PN_J0633_15asec_grp15.pi 2:2 PN_J0633_15asec_bkg.pi 3:3 M1_J0633_15asec_grp15.pi 4:4 M1_J0633_bkg.pi 5:5 M2_J0633_15asec_grp15.pi 6:6 M2_J0633_15asec_bkg.pi 7:7 PN_pwn_ex_grp15.pi 8:8 PN_pwn_ex_bkg.pi 9:9 M1_pwn_ex_grp15.pi 10:10 M1_pwn_ex_bkg.pi 11:11 M2_pwn_ex_grp15.pi 12:12 M2_pwn_ex_bkg.pi"

nspec = 12

AllData(SPECNAME)

for i in range(nspec):
    AllData(i+1).background = " "

AllData.ignore("**-%2.1f %2.1f-**"%(erange[0],erange[1]))
AllData.ignore("bad")

scl = [288000. / 2241600., 271732. / 2207424., 286400. / 2241600., 2595200. / 2241600., 2574576. / 2207424., 2465192. / 2241600.]

pars = read_data(sys.argv[2])
print pars.shape

Mns = 1.4
Rns = 13.

nh = pars[1,12]
Teff = pars[1,0]
logR = math.log10(Rns)
logN = pars[1,1]
mgfld = float(sys.argv[3])
logD = pars[1,13] - 3.0
psrIndx = pars[1,2]
psrNrm = pars[1,3]
pwnIndx = pars[1,4]
pwnNrm = pars[1,5]

bckIndx = [pars[1,6], pars[1,8], pars[1,10], 0.88, 1.12, 1.13 ]
bckNrm = [pars[1,7], pars[1,9], pars[1,11], -5.00, -5.08, -5.05]

"""
AllModels += "(nsa+powerlaw)*phabs+powerlaw"
for i in range(int(nspec/2./2.)):
    AllModels(2*i+1).setPars((Teff, Mns, 10**logR, mgfld, 10**(2.*logN)*10**(-2.*logD), psrIndx, 10**psrNrm, nh, bckIndx[i], scl[i]*10**bckNrm[i]))
    AllModels(2*i+2).setPars((Teff, Mns, 10**logR, mgfld, 0., psrIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
    AllModels(2*i+1+int(nspec/2.)).setPars((Teff, Mns, 10**logR, mgfld, 0., pwnIndx, 10**pwnNrm, nh, bckIndx[i], scl[i+int(nspec/2./2.)]*10**bckNrm[i]))
    AllModels(2*i+2+int(nspec/2.)).setPars((Teff, Mns, 10**logR, mgfld, 0., pwnIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
"""
AllModels += "(nsmaxg+powerlaw)*phabs+powerlaw"
for i in range(int(nspec/2./2.)):
    AllModels(2*i+1).setPars((Teff, Mns, 10**logR, 10**logD, int(mgfld), 10**(2*logN), psrIndx, 10**psrNrm, nh, bckIndx[i], scl[i]*10**bckNrm[i]))
    AllModels(2*i+2).setPars((Teff, Mns, 10**logR, 10**logD, int(mgfld), 0., psrIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
    AllModels(2*i+1+int(nspec/2.)).setPars((Teff, Mns, 10**logR, 10**logD, int(mgfld), 0., pwnIndx, 10**pwnNrm, nh, bckIndx[i], scl[i+int(nspec/2./2.)]*10**bckNrm[i]))
    AllModels(2*i+2+int(nspec/2.)).setPars((Teff, Mns, 10**logR, 10**logD, int(mgfld), 0., pwnIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
"""
AllModels += "(bbodyrad+powerlaw)*phabs+powerlaw"
for i in range(int(nspec/2./2.)):
    AllModels(2*i+1).setPars((Teff, 1.E8*10**(2*logR), psrIndx, 10**psrNrm, nh, bckIndx[i], scl[i]*10**bckNrm[i]))
    AllModels(2*i+2).setPars((Teff, 0., psrIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
    AllModels(2*i+1+int(nspec/2.)).setPars((Teff, 0., pwnIndx, 10**pwnNrm, nh, bckIndx[i], scl[i+int(nspec/2./2.)]*10**bckNrm[i]))
    AllModels(2*i+2+int(nspec/2.)).setPars((Teff, 0., pwnIndx, 0., nh, bckIndx[i], 10**bckNrm[i]))
"""
Fit.show()
AllModels.show()
print Fit.statistic
Fit.show()

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

Plot("delchi")

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

ax[0] = plt.subplot(gs[:5,0])
ax[1] = plt.subplot(gs[5:10,0])
ax[2] = plt.subplot(gs[10:12,0])
ax[3] = plt.subplot(gs[12:14,0])
axax = fig.add_subplot(gs[:10,0],frameon=False)
axwx = fig.add_subplot(gs[10:14,0],frameon=False)

setcolours = ['b','g','r','b','g','r']

if ( psr == 1 ):
    for i in range(int(nspec/2./2.)):
        ax[0].errorbar(spcx[2*i],spcy[2*i],xerr=spcrrx[2*i],yerr=spcrry[2*i],color=setcolours[i],fmt=' ',capsize=0)
        ax[0].step(np.append(spcx[2*i][0]-spcrrx[2*i][0],spcx[2*i]+spcrrx[2*i]),np.append(mod[2*i][0],mod[2*i]),color=setcolours[i])
        ax[1].errorbar(spcx[2*i+1],scl[i]*spcy[2*i+1],xerr=spcrrx[2*i+1],yerr=scl[i]*spcrry[2*i+1],color=setcolours[i],fmt=' ',capsize=0)
        ax[1].step(np.append(spcx[2*i+1][0]-spcrrx[2*i+1][0],spcx[2*i+1]+spcrrx[2*i+1]),np.append(scl[i]*mod[2*i+1][0],scl[i]*mod[2*i+1]),color=setcolours[i])
        ax[2].errorbar(spcx[2*i],chiy[2*i],xerr=spcrrx[2*i],yerr=chirry[2*i],color=setcolours[i],fmt=' ',capsize=0)
        ax[3].errorbar(spcx[2*i+1],scl[i]*chiy[2*i+1],xerr=spcrrx[2*i+1],yerr=scl[i]*chirry[2*i+1],color=setcolours[i],fmt=' ',capsize=0)
    xqu1 = [0.1,erange2[0],erange2[0],0.1]
    xqu2 = [erange2[1],12.,12.,erange2[1]]
    spcnum = 2
    spcminy = spcy[2*spcnum]-spcrry[2*spcnum]
    spcnum = 0
    spcmaxy = spcy[2*spcnum]+spcrry[2*spcnum]
    yqu = [spcminy.min(),spcminy.min(),spcmaxy.max()+1.*(spcmaxy.max()-spcminy.min()),spcmaxy.max()+1.*(spcmaxy.max()-spcminy.min())]
    ax[0].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[0].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[0].set_ylim(spcminy.min()+7.e-6,spcmaxy.max()+1.*(spcmaxy.max()-spcminy.min()))
    ax[0].text(2.2,5.e-2,'psr+background')
    spcnum = 2
    spcminy = spcy[2*spcnum+1]-spcrry[2*spcnum+1]
    spcnum = 0
    spcmaxy = spcy[2*spcnum+1]+spcrry[2*spcnum+1]
    yqu = [spcminy.min(),spcminy.min(),spcmaxy.max()+3.*(spcmaxy.max()-spcminy.min()),spcmaxy.max()+3.*(spcmaxy.max()-spcminy.min())]
    ax[1].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[1].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[1].set_ylim(spcminy.min()+9.e-7,6.e-2)
    ax[1].text(2.2,8.e-3,'background')
    yqu = [-5.,-5.,5.,5.]
    ax[2].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[2].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[2].set_ylim(-4.5,4.5)
    yqu = [-.5,-.5,.5,.5]
    ax[3].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[3].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[3].set_ylim(-.48,.48)
    l = ax[0].legend(['pn','MOS1','MOS2'],fontsize=9,loc=(0.2,0.05))
    l.set_zorder(5)
else:
    for i in range(int(nspec/2./2.)):
        ax[0].errorbar(spcx[2*i+int(nspec/2.)],spcy[2*i+int(nspec/2.)],xerr=spcrrx[2*i+int(nspec/2.)],yerr=spcrry[2*i+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
        ax[0].step(np.append(spcx[2*i+int(nspec/2.)][0]-spcrrx[2*i+int(nspec/2.)][0],spcx[2*i+int(nspec/2.)]+spcrrx[2*i+int(nspec/2.)]),np.append(mod[2*i+int(nspec/2.)][0],mod[2*i+int(nspec/2.)]),color=setcolours[i+int(nspec/2./2.)])
        ax[1].errorbar(spcx[2*i+1+int(nspec/2.)],scl[i+int(nspec/2./2.)]*spcy[2*i+1+int(nspec/2.)],xerr=spcrrx[2*i+1+int(nspec/2.)],yerr=scl[i+int(nspec/2./2.)]*spcrry[2*i+1+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
        ax[1].step(np.append(spcx[2*i+1+int(nspec/2.)][0]-spcrrx[2*i+1+int(nspec/2.)][0],spcx[2*i+1+int(nspec/2.)]+spcrrx[2*i+1+int(nspec/2.)]),np.append(scl[i+int(nspec/2./2.)]*mod[2*i+1+int(nspec/2.)][0],scl[i+int(nspec/2./2.)]*mod[2*i+1+int(nspec/2.)]),color=setcolours[i+int(nspec/2./2.)])
        ax[2].errorbar(spcx[2*i+int(nspec/2.)],chiy[2*i+int(nspec/2.)],xerr=spcrrx[2*i+int(nspec/2.)],yerr=chirry[2*i+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
        ax[3].errorbar(spcx[2*i+1+int(nspec/2.)],scl[i+int(nspec/2./2.)]*chiy[2*i+1+int(nspec/2.)],xerr=spcrrx[2*i+1+int(nspec/2.)],yerr=scl[i+int(nspec/2./2.)]*chirry[2*i+1+int(nspec/2.)],color=setcolours[i+int(nspec/2./2.)],fmt=' ',capsize=0)
    xqu1 = [0.1,erange2[0],erange2[0],0.1]
    xqu2 = [erange2[1],12.,12.,erange2[1]]
    spcnum = 2
    spcminy = spcy[2*spcnum+int(nspec/2.)]-spcrry[2*spcnum+int(nspec/2.)]
    spcnum = 0
    spcmaxy = spcy[2*spcnum+int(nspec/2.)]+spcrry[2*spcnum+int(nspec/2.)]
    yqu = [spcminy.min(),spcminy.min(),spcmaxy.max()+3.*(spcmaxy.max()-spcminy.min()),spcmaxy.max()+3.*(spcmaxy.max()-spcminy.min())]
    ax[0].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[0].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[0].set_ylim(spcminy.min()+1.e-5,2.e-1)
    ax[0].text(2.2,5.e-2,'pwn+background')
    spcnum = 2
    spcminy = spcy[2*spcnum+1+int(nspec/2.)]-spcrry[2*spcnum+1+int(nspec/2.)]
    spcnum = 0
    spcmaxy = spcy[2*spcnum+1+int(nspec/2.)]+spcrry[2*spcnum+1+int(nspec/2.)]
    yqu = [spcminy.min(),spcminy.min(),spcmaxy.max()+.1*(spcmaxy.max()-spcminy.min()),spcmaxy.max()+1.*(spcmaxy.max()-spcminy.min())]
    ax[1].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[1].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[1].set_ylim(spcminy.min()+1.e-5,3.e-1)
    ax[1].text(2.2,4.e-2,'background')
    yqu = [-5.,-5.,5.,5.]
    ax[2].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[2].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[2].set_ylim(-4.5,4.5)
    yqu = [-5.,-5.,5.,5.]
    ax[3].fill(xqu1,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[3].fill(xqu2,yqu,color='0.4',alpha=0.6, zorder=4)
    ax[3].set_ylim(-4.8,4.8)
    l = ax[0].legend(['pn','MOS1','MOS2'],fontsize=9,loc=(0.35,0.05))
    l.set_zorder(5)

ax[2].plot(erange,[0.0,0.0],'--',color='k')
ax[3].plot(erange,[0.0,0.0],'--',color='k')

subs = [1.0, 2.0, 4.0, 7.0]

for i in range(4):
    ax[i].set_xlim(erange[0],erange[1])
    ax[i].set_xscale('log')
    #ax[i].xaxis.set_major_formatter(CustomTicker())
    ax[i].xaxis.set_minor_locator(ticker.LogLocator(subs=subs)) #set the ticks position
    ax[i].xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
    ax[i].tick_params(axis='both', which='major', labelsize=10)
    ax[i].tick_params(axis='both', which='minor', labelsize=10)

ax[3].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  #add the custom ticks

plt.setp(ax[3].get_xticklabels(minor=True), visible=True)

for i in range(2):
    ax[i].set_yscale('log',nonposy='clip')
    #ax[i].yaxis.set_major_formatter(LogFormatterSciNotation())

plt.setp([a.get_xticklabels() for a in ax[:3]], visible=False)
plt.setp([a.get_xticklabels(minor=True) for a in ax[:3]], visible=False)

#ax[i].set_ylabel(r'$\rm normalized \, counts \, s^{-1} \, keV^{-1} $',fontsize=10)
ax[3].set_xlabel(r'$ \rm Photon \, energy  \, [\, \rm keV\,] $',fontsize=10)
axax.set_ylabel(r'$ \rm normalized \, counts \, s^{-1} \, keV^{-1} $',fontsize=10,labelpad=35)
axwx.set_ylabel(r'$ \chi $',fontsize=10,labelpad=35)
#axax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.setp(axax.get_xticklabels(), visible=False)
plt.setp(axax.get_yticklabels(), visible=False)
plt.setp(axwx.get_xticklabels(), visible=False)
plt.setp(axwx.get_yticklabels(), visible=False)
axax.set_xticks([])
axax.set_yticks([])
axwx.set_xticks([])
axwx.set_yticks([])

#ax[3].set_ylabel(r'$ \rm sign(data-model)\Delta\chi^{2} $',fontsize=10)

if psr == 1:
    plt.savefig("psrspec"+sys.argv[2]+".pdf")
else:
    plt.savefig("pwnspec"+sys.argv[2]+".pdf")
#plt.show()
