#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *
import asciitable
from xspec import *
import emcee
from scipy.interpolate import UnivariateSpline

NameChain  = sys.argv[1]

firstrun   = sys.argv[2]
thread_num = float(sys.argv[3])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"% (thread_num)+".dat"
LastPos    = NameChain+"_"+"LastPos"+".dat"

whatatm = 1200

Mns      = 1.4
Rns      = 13.
Bns      = 1.E12
redshift = (1-2.952*(Mns/Rns))**(-1./2.)

Fit.statMethod = "chi"
Fit.statTest   = "chi"

#SPECNAME = "1:1 Calvera-PN-1_snr3.pi 1:2 Calvera-PN-2_snr3.pi 2:3 Calvera-M1-1_snr3.pi 2:4 Calvera-M1-2_snr3.pi 3:5 calvera-acis-cc2_grp.pi 3:6 calvera-acis-cc1_grp.pi 4:7 calvera-acis-s_grp.pi"
SPECNAME = "1:1 Calvera-PN-1_min25.pi 1:2 Calvera-PN-2_min25.pi 2:3 Calvera-M1-1_min25.pi 2:4 Calvera-M1-2_min25.pi 3:5 calvera-acis-cc2_grp25.pi 3:6 calvera-acis-cc1_grp25.pi 4:7 calvera-acis-s_grp25.pi"

ignore_lessXMM = "**-0.3"
ignore_moreXMM = "7.0-**"
ignore_lessCC  = "**-0.5"
ignore_moreCC  = "7.0-**"
ignore_less    = "**-0.5" 
ignore_more    = "7.0-**"   

AllData(SPECNAME)
for i in range(1,5):
    AllData(i).ignore(ignore_lessXMM)
    AllData(i).ignore(ignore_moreXMM)
for i in range(5,7):
    AllData(i).ignore(ignore_lessCC)
    AllData(i).ignore(ignore_moreCC)  
for i in range(7,8):
    AllData(i).ignore(ignore_less)
    AllData(i).ignore(ignore_more)
AllData.ignore("bad")

AllModels += "tbabs*(nsmax+powerlaw)*gabs"

Xset.abund = "wilm"
Xset.xsect = "bcmc"
print Xset.abund
print Xset.xsect

AllData.show()
AllModels.show()
Fit.show()
Xset.chatter = 0

ndim     = 12
nsteps   = 10000
nwalkers = 100

def lnprior(theta):
    nh,T,Z,E0,lsigma,ltau,lRD1,lRD2,lRD3,lRD4,PhoInd,logPLNorm = tuple(theta)
    if 0.0<nh and 5.5<T<6.8 and 0.6<E0<1. and -2.6<lsigma<-0.5 and -3.<ltau<6. and 1.1<Z<2.0 and logPLNorm<24. and 0.5<PhoInd<9.0:
        return logPLNorm
    return -np.inf

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    for i in range(4):
        AllModels(i+1).setPars((theta[0],theta[1],theta[2],whatatm,10**(2.*theta[i+6]),theta[10],10**theta[11],theta[3],10**theta[4],10**theta[5]))
    fs=Fit.statistic
    return lp - 0.5*fs

parsinit = (0.015,6.0,1.2,0.74,-0.99,0.,0.7,0.7,0.7,0.7,2.0,-6.)
pos      = [parsinit + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

if firstrun == "N":
    inf   = open(LastPos)
    lines = inf.readlines()  
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    inf.close()  
        
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(),threads=10)

nsteps_per=nsteps/100
sys.stdout.write('[0%')
for k in range(92):
    sys.stdout.write(' ')
sys.stdout.write('100%]')
print
for k in range(100):
    if firstrun == "N":
        result = sampler.run_mcmc(prepos,nsteps_per)
        prepos=result[0]
    elif firstrun == "Y":
        result = sampler.run_mcmc(pos,nsteps_per)
        pos=result[0]
    sys.stdout.write('#')
    sys.stdout.flush()
    lastpos = result[0]
    f = open(LastPos, "w")
    n1, n2 = shape(lastpos)
    for i in range(n1):
        for j in range(n2):
            f.write("%.15E "%(lastpos[i,j]))
        f.write("\n")
    f.close()

lastpos = result[0]
f = open(LastPos, "w")
n1, n2 = shape(lastpos)
for i in range(n1):
    for j in range(n2):
        f.write("%.15E "%(lastpos[i,j]))
    f.write("\n")
f.close()

print "Autocorelation time:"
print sampler.acor
print "Acceptance fraction:"
print np.amax(sampler.acceptance_fraction)
print np.argmax(sampler.acceptance_fraction)
print np.amin(sampler.acceptance_fraction)
print np.argmin(sampler.acceptance_fraction)

steps = linspace(1,nsteps,num=nsteps)
fig, ax = plt.subplots(nrows=ndim)
plt.subplots_adjust(hspace=0.1)
for i in range(ndim):
    for j in range(nwalkers):
        ax[i].errorbar(steps,sampler.chain[j,:,i])
setp([a.get_xticklabels() for a in ax[:ndim-1]], visible=False)
#plt.show()

burn = 0 #int(raw_input("How many steps to discard: "))

samples = sampler.chain[:,burn:,:].reshape((-1, ndim))
likely  = sampler.lnprobability[:,burn:].reshape((-1))

f = open(SampleName, "w")
n1, n2 = shape(samples)
for i in range(n1):
    for j in range(n2):
        f.write("%.15E "%(samples[i,j]))
    f.write("%.15E "%(likely[i] ))  
    f.write("\n")
f.close()
