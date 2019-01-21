#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
from pylab import *
from astropy.io import ascii
import numpy as np
import emcee
import scipy.optimize as op
from StringIO import StringIO

pi = 3.14159265358979323846

Crds_file = sys.argv[1]

NameChain  = sys.argv[2]
firstrun   = sys.argv[3]
thread_num = float(sys.argv[4])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"% (thread_num)+".dat"
LastPos    = NameChain+"_"+"LastPos"+".dat"

nsteps   = 1000
nwalkers = 1000

def parse_input(in_file):
    f = open(in_file)
    lines = f.readlines()  
    n = len(lines)
    #print n
    coords = []
    for i in range(n):
        coords.append([float(x) for x in lines[i].split()])
    coords_1 = np.empty([len(coords),6])
    for i in range(len(coords)):
        coords_1[i] = np.array([coords[i]])  
    return coords_1

coords = parse_input(Crds_file)

coords_1 = np.array([coords[:4],coords[4:8],coords[8:12]]) 
#print coords_1
print len(coords_1)
print shape(coords_1)

N_im = len(coords_1)
N_stars = len(coords_1[0])

cosdelta  = cos((pi/180.)*coords_1[0,3,2])
raOff     = (coords_1[:,:,1] - coords_1[0,3,1])*3600*cosdelta
decOff    = (coords_1[:,:,2] - coords_1[0,3,2])*3600
errraOff  = coords_1[:,:,3]*3600*cosdelta
errdecOff = coords_1[:,:,4]*3600
xref      = raOff+1j*decOff
    
def lnprob(th):
    trsf = np.empty([N_im,3])
    trsf[0,:] = np.array([(0.,0.,0.)])
    for i in range(1,N_im):
        trsf[i,:] = np.array([(th[3*i],th[3*i+1],th[3*i+2])])
    xs1 = np.empty([N_stars,2])
    for i in range(N_stars):
        xs1[i,0] = th[3*(N_im-1)+2*i]
        xs1[i,1] = th[3*(N_im-1)+2*i+1]
    xs   = xs1[:,0]+1j*xs1[:,1] 
    pm = np.empty([N_im,2])
    pm[0,:] = np.array([(0.,0.)])
    for i in range(1,N_im):
        pm[i,:] = np.array([(th[-2],th[-1])])
    prob = np.empty([N_im,N_stars])
    for i in range(N_im):
        for k in range(N_stars):
            if k == 3:
                delx      = pm[i,0]*exp(1j*pm[i,1])+trsf[i,0]+1j*trsf[i,1]+exp(1j*trsf[i,2])*xs[k] - xref[i,k]
                prob[i,k] = (delx.real/errraOff[i,k])**2 + (delx.imag/errdecOff[i,k])**2
            else:
                delx      = trsf[i,0]+1j*trsf[i,1]+exp(1j*trsf[i,2])*xs[k] - xref[i,k]
                prob[i,k] = (delx.real/errraOff[i,k])**2 + (delx.imag/errdecOff[i,k])**2
    return -0.5*prob.sum()

print raOff
print decOff

#exit()
p0=[]
for i in range(N_im-1):
    p0.append(0.)
    p0.append(0.)
    p0.append(0.)
for j in range(N_stars):
    p0.append(raOff[0,j])
    p0.append(decOff[0,j])
p0.append(0.)
p0.append(0.)

print len(p0)

ndim     = 3*(N_im-1) + 2 + 2*N_stars
print ndim
#exit()
pos = [p0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

if firstrun == "N":
    inf   = open(LastPos)
    lines = inf.readlines()  
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    inf.close()  

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(), threads=10)

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
#print sampler.acor
print "Acceptance fraction:"
print np.amax(sampler.acceptance_fraction)
print np.argmax(sampler.acceptance_fraction)
print np.amin(sampler.acceptance_fraction)
print np.argmin(sampler.acceptance_fraction)

nPlot = 5

steps = linspace(1,nsteps,num=nsteps)
fig, ax = plt.subplots(nrows=nPlot)
plt.subplots_adjust(hspace=0.1)
for i in range(nPlot):
    for j in range(nwalkers):
        ax[i].errorbar(steps,sampler.chain[j,:,i])
setp([a.get_xticklabels() for a in ax[:nPlot-1]], visible=False)
plt.show()

burn = int(raw_input("How many steps to discard: "))

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
