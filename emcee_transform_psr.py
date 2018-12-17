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

NameChain  = sys.argv[1]

firstrun   = sys.argv[2]
thread_num = float(sys.argv[3])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"% (thread_num)+".dat"
LastPos    = NameChain+"_"+"LastPos"+".dat"

pi   = 3.141592654

Cords1      = sys.argv[4]
Cords2      = sys.argv[5]
#psr_coords1 = sys.argv[6]
#psr_coords2 = sys.argv[7]

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

coords1 = parse_input(Cords1)
coords2 = parse_input(Cords2)
psr1 = coords1[-1:,:] #parse_input(psr_coords1)
psr2 = coords2[-1:,:] #parse_input(psr_coords2)
coords1 = coords1[:-1,:]
coords2 = coords2[:-1,:]
print coords1
#exit()
print psr1
print psr2

Num_sources = len(coords1)
print len(coords1),len(coords2)
print len(psr1),len(psr2)
#exit()

ndim     = Num_sources*2+3+2+4
#ndim     = Num_sources*2+3+2
nsteps   = 3000
nwalkers = 500

def lnprob(theta,coords1,coords2):
    z0  = theta[0]+1j*theta[1]
    phi = theta[2]
    s1  = theta[3]
    s2  = theta[4]
    Ns = len(coords1)
    xsx = np.empty([Ns])
    xsy = np.empty([Ns])
    for i in range(Ns):
        xsx[i] = theta[2*i+5]
        xsy[i] = theta[2*i+6]
    xs = xsx+1j*xsy
    x1 = coords1[:,0]+1j*coords1[:,1]
    x2 = coords2[:,0]+1j*coords2[:,1]
    delx1 = xs-x1
    delx2 = (exp(1j*phi)*xs+z0)
    delx2 = delx2.real*s1+1j*delx2.imag*s2 - x2
    rotdelx1 = delx1*exp(1j*coords1[:,4])
    rotdelx2 = delx2*exp(1j*coords2[:,4])
    prob1 = (rotdelx1.real/coords1[:,2])**2 + (rotdelx1.imag/coords1[:,3])**2
    prob2 = (rotdelx2.real/coords2[:,2])**2 + (rotdelx2.imag/coords2[:,3])**2
    prob  = prob1 + prob2
    return -0.5*prob.sum()

def lnprob_psr(theta,coords1,coords2,psr1,psr2):
    z0  = theta[0]+1j*theta[1]
    phi = theta[2]
    s1  = theta[3]
    s2  = theta[4]
    xp  = theta[-4]+1j*theta[-3]
    xpm = theta[-2]+1j*theta[-1]
    xp1 = psr1[0,0]+1j*psr1[0,1]
    xp2 = psr2[0,0]+1j*psr2[0,1]
    delxp1    = xp-xp1
    delxp2    = (exp(1j*phi)*(xp+xpm)+z0)
    delxp2    = delxp2.real*s1+1j*delxp2.imag*s2-xp2
    rotdelxp1 = delxp1*exp(1j*psr1[0,4])
    rotdelxp2 = delxp2*exp(1j*psr2[0,4])
    probpsr1  = (rotdelxp1.real/psr1[0,2])**2 + (rotdelxp1.imag/psr1[0,3])**2
    probpsr2  = (rotdelxp2.real/psr2[0,2])**2 + (rotdelxp2.imag/psr2[0,3])**2
    probpsr   = probpsr1 + probpsr2
    return lnprob(theta,coords1,coords2)-0.5*probpsr

p0 = []
p0 = [0.,0.,0.,1.,1.]
for i in range(Num_sources):
    p0.append(coords1[i,0]*(0.995+1e-2*np.random.rand()))
    p0.append(coords1[i,1]*(0.995+1e-2*np.random.rand()))  
p0.append(psr1[0][0]*(0.995+1e-2*np.random.rand()))  
p0.append(psr1[0][1]*(0.995+1e-2*np.random.rand()))
p0.append(0.)
p0.append(0.)

pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
if firstrun == "N":
    inf   = open(LastPos)
    lines = inf.readlines()  
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    inf.close()  
        
#sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(),threads=10)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_psr, args=(coords1,coords2,psr1,psr2), threads=10)

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

