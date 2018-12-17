#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
from pylab import *
import numpy as np
import emcee
import scipy.optimize as op
from StringIO import StringIO
from astropy import units as u
from astropy.coordinates import SkyCoord

NameChain = sys.argv[1]

firstrun = sys.argv[2]
thread_num = float(sys.argv[3])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"% (thread_num)+".dat"
LastPos = NameChain+"_"+"LastPos"+".dat"

pi = 3.141592654

Coords = sys.argv[4]

def parse_input(in_file):
  f = open(in_file)
  lines = []
  for l in f:
    lines.append(str(l))
  coords = []
  for i in range(len(lines)):
    try:
      #print lines[i].split()
      if lines[i].split()[0] == 'Offset':
        offx,offy  = tuple(lines[i].split())[3:5]
      elif  lines[i].split()[0] == 'Pos':
        errelmaj,errelmin,errelang = tuple(lines[i].split())[4:7]
      elif lines[i].split()[0] == 'Right':
        ra = lines[i].split()[2]
      elif lines[i].split()[0] == 'Declination:':
        dec = lines[i].split()[1]
        cc = SkyCoord(ra+' '+dec, frame='icrs', unit=(u.hourangle, u.deg))
        coords.append((float(offx),float(offy),float(errelmaj),float(errelmin),float(errelang),float(cc.ra.degree),float(cc.dec.degree)))
    except:
      pass
    coords_1 = np.empty([len(coords),7])
    for i in range(len(coords)):
      coords_1[i] = np.array([coords[i]])
  return coords_1

coords = parse_input(Coords)
Tepoch = [0.,6.65479,11.48219]

NumEpoch = len(coords)
print NumEpoch

ndim = 4

nsteps = 500
nwalkers = 500

def lnprob(theta,coords,Tepoch):
    xp = theta[0]+1j*theta[1]
    pm = theta[2]+1j*theta[3]
    NumEpoch = len(coords)
    cosdelta = cos((pi/180.)*coords[0,6])
    prob = np.empty([NumEpoch])
    for i in range(NumEpoch):
        raOff = (coords[i,5] - coords[0,5])*3600*cosdelta
        decOff = (coords[i,6] - coords[0,6])*3600
        xp0 = raOff+1j*decOff
        delxp = (xp+pm*Tepoch[i])-xp0
        rotdelxp = delxp*exp(1j*coords[i,4])
        prob[i] = (rotdelxp.real/coords[i,2])**2 + (rotdelxp.imag/coords[i,3])**2
    return -0.5*prob.sum()

p0 = [0.,0.,0.,0.]

pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

if firstrun == "N":
    inf = open(LastPos)
    lines = inf.readlines()
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    inf.close()

#sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(),threads=10)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(coords,Tepoch), threads=10)

nsteps_per = nsteps/100
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
        pos = result[0]
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
plt.show()

burn = 0 #int(raw_input("How many steps to discard: "))

samples = sampler.chain[:,burn:,:].reshape((-1, ndim))
likely = sampler.lnprobability[:,burn:].reshape((-1))

f = open(SampleName, "w")
n1, n2 = shape(samples)
for i in range(n1):
    for j in range(n2):
        f.write("%.15E "%(samples[i,j]))
    f.write("%.15E "%(likely[i] ))
    f.write("\n")
f.close()
