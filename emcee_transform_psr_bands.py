#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from astropy.io import ascii
import emcee
import scipy.optimize as op
from StringIO import StringIO

pi = 3.14159265358979323846

def ParseInput(infile):
    f = open(infile)
    lines = []
    for l in f:
        lines.append(str(l))
    coords = []
    source_num_old=0
    for i in range(len(lines)):
        try:
            ls=lines[i].split()
            if ls[0].strip() == 'Source':
                if ls[1].strip(',') != "PSR":
                    source_num = float(ls[1].strip(','))
                else:
                    source_num = 55
            elif ls[0] == 'Offset':
                offx,offy  = tuple(ls)[3:5]
            elif ls[0] == 'Pos':
                errelmaj,errelmin,errelang = tuple(ls)[4:7]
                if source_num != source_num_old:
                    coords.append((float(offx),float(offy),float(errelmaj),float(errelmin),(pi/180.)*float(errelang)))
                    source_num_old = source_num
                else:
                    coords[-1] = tuple(float(offx),float(offy),float(errelmaj),float(errelmin),(pi/180.)*float(errelang))
        except:
            pass
    coords_1 = np.empty([len(coords),5])
    for i in range(len(coords)):
        coords_1[i] = np.array([coords[i]])
    return coords_1

def FoldInput(infile):
    inf = open(infile)
    lines = inf.readlines()
    epochs = np.zeros([len(lines)])
    epoch[0] = lines[0].split()[1]
    coords_array = np.array([ParseInput(lines[0].split()[0])])
    for i in range(len(lines)-1):
        coords_array = np.append(coords_array,np.array([ParseInput(lines[i+1].split()[0])]),axis=0)
        epochs[i+1] = lines[i+1].split()[1]
    return coords_array, epochs

imgs = sys.argv[1]
NameChain = sys.argv[2]
firstrun = sys.argv[3]
thread_num = float(sys.argv[4])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"% (thread_num)+".dat"
LastPos = NameChain+"_"+"LastPos"+".dat"

nsteps = 200
nwalkers = 500

crds,epchs = FoldInput(imgs)
crdsref = crds[:,:-1,:]
crdspsr = crds[:,-1:,:]

nim = shape(crdsref)[0]
nst = shape(crdsref)[1]
print nim, nst
print epochs
nref = 0

exit()

def lnprob(th):
    ind = []
    for i in range(nim):
        if i != nref:
            ind.append(i)
    trsf = np.empty([nim,5])
    trsf[nref,:] = np.array([(0.,0.,0.,1.,1.)])
    for i in range(nim-1):
        trsf[ind[i],:] = np.array([(th[5*i],th[5*i+1],th[5*i+2],th[5*i+3],th[5*i+4])])
    xs1 = np.empty([nst])
    xs2 = np.empty([nst])
    for i in range(nst):
        xs1[i] = th[5*(nim-1)+2*i]
        xs2[i] = th[5*(nim-1)+2*i+1]
    xs = xs1+1j*xs2
    xref = crdsref[:,:,0]+1j*crdsref[:,:,1]
    prob = np.empty([nim,nst])
    for i in range(nim):
        delx = exp(1j*trsf[i,2])*xs+trsf[i,0]+1j*trsf[i,1]
        delx = delx.real*trsf[i,3]+1j*delx.imag*trsf[i,4] - xref[i,:]
        rotdelx = delx*exp(1j*crdsref[i,:,4])
        prob[i,:] = (rotdelx.real/crdsref[i,:,2])**2 + (rotdelx.imag/crdsref[i,:,3])**2
    return -0.5*prob.sum()

def lnprob_psr(th):
    ind = []
    for i in range(nim):
        if i != nref:
            ind.append(i)
    trsf = np.empty([nim,5])
    trsf[nref,:] = np.array([(0.,0.,0.,1.,1.)])
    for i in range(nim-1):
        trsf[ind[i],:] = np.array([(th[5*i],th[5*i+1],th[5*i+2],th[5*i+3],th[5*i+4])])
    starsdim = 5*(nim-1)+2*nst
    xp = th[starsdim]+1j*th[starsdim+1]
    xpm = th[starsdim+2]+1j*th[starsdim+3]
    xp_ref = crdspsr[:,0,0]+1j*crdspsr[:,0,1]
    pm_im = np.zeros([nim],dtype=complex)
    for i in nim:
        pm_im[i] = xpm * (epchs[i] - epchs[nref]) / 365
    probpsr = np.empty([nim])
    for i in range(nim):
        delxp = exp(1j*trsf[i,2])*(xp+pm_im[i])+trsf[i,0]+1j*trsf[i,1]
        delxp = delxp.real*trsf[i,3]+1j*delxp.imag*trsf[i,4] - xp_ref[i]
        rotdelxp = delxp*exp(1j*crdspsr[i,0,4])
        probpsr[i] = (rotdelxp.real/crdspsr[i,0,2])**2 + (rotdelxp.imag/crdspsr[i,0,3])**2
    return lnprob(th)-0.5*probpsr.sum()

p0=[]
for i in range(nim-1):
    p0.append(0.)
    p0.append(0.)
    p0.append(0.)
    p0.append(1.)
    p0.append(1.)
for j in range(nst):
    p0.append(coords_ref[nref,j,0]*(0.995+1e-2*np.random.rand()))
    p0.append(coords_ref[nref,j,1]*(0.995+1e-2*np.random.rand()))
p0.append(coords_psr[nref,0,0]*(0.995+1e-2*np.random.rand()))
p0.append(coords_psr[nref,0,1]*(0.995+1e-2*np.random.rand()))
p0.append(0.)
p0.append(0.)

ndim = 5*(nim-1)+2*nst + 4

pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

if firstrun == "N":
    inf = open(LastPos)
    lines = inf.readlines()
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    inf.close()

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob_psr,args=(),threads=10)

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

nPlot = 5

steps = linspace(1,nsteps,num=nsteps)
fig, ax = plt.subplots(nrows=nPlot)
plt.subplots_adjust(hspace=0.1)
for i in range(nPlot):
    for j in range(nwalkers):
        ax[i].errorbar(steps,sampler.chain[j,:,i])
setp([a.get_xticklabels() for a in ax[:nPlot-1]], visible=False)
#plt.show()
plt.savefig(NameChain+".pdf")

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
