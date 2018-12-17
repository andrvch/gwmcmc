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

Crds_files = sys.argv[1]

NameChain = sys.argv[2]

firstrun = sys.argv[3]
thread_num = float(sys.argv[4])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"% (thread_num)+".dat"
LastPos = NameChain+"_"+"LastPos"+".dat"

#withpsr = sys.argv[6]

nsteps = 10000
nwalkers = 500

def parse_input(in_file):
    f = open(in_file)
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

def fold_input(in_file):
    inf   = open(in_file)
    lines = inf.readlines()
    coords_array = np.array([parse_input(lines[0].strip())])
    for i in range(len(lines)-1):
        coords_array = np.append(coords_array,np.array([parse_input(lines[i+1].strip())]),axis=0)
    return coords_array

coords = fold_input(Crds_files)

coords_ref = coords[:,:-1,:]
coords_psr = coords[:,-1:,:]

print shape(coords_ref)
print shape(coords_psr)
print shape(coords_ref)[0]

N_im = shape(coords_ref)[0]
N_stars = shape(coords_ref)[1]

Nm_ref = 1
Nm_epc1 = [4,5,6,7]
Nm_epc2 = [8]

DeltaEpc1 =
DeltaEpc2 =

def lnprob(th):
    N_im = len(coords)
    trsf = np.empty([N_im,5])
    ind = []
    for i in range(N_im):
        if i != Nm_ref:
            ind.append(i)
    for i in range(N_im-1):
        trsf[ind[i],:] = np.array([(th[5*i],th[5*i+1],th[5*i+2],th[5*i+3],th[5*i+4])])
    trsf[Nm_ref,:] = np.array([(0.,0.,0.,1.,1.)])
    N_stars = len(coords[0])
    xs1 = np.empty([N_stars])
    xs2 = np.empty([N_stars])
    for i in range(N_stars):
        xs1[i] = th[5*(N_im-1)+2*i]
        xs2[i] = th[5*(N_im-1)+2*i+1]
    xs = xs1+1j*xs2
    xref = coords[:,:,0]+1j*coords[:,:,1]
    prob = np.empty([N_im,N_stars])
    for i in range(N_im):
        delx = (exp(1j*trsf[i,2])*xs+trsf[i,0]+1j*trsf[i,1])
        delx = delx.real*trsf[i,3]+1j*delx.imag*trsf[i,4] - xref[i,:]
        rotdelx = delx*exp(1j*coords[i,:,4])
        prob[i,:] = (rotdelx.real/coords[i,:,2])**2 + (rotdelx.imag/coords[i,:,3])**2
    return -0.5*prob.sum()

def lnprob_psr(th):
    N_im = shape(coords_ref)[0]
    N_stars = shape(coords_ref)[1]
    starsdim = 5*(N_im-1)+2*N_stars
    trsf = np.empty([N_im,5])
    ind = []
    for i in range(N_im):
        if i != Nm_ref:
            ind.append(i)
    for i in range(N_im-1):
        trsf[ind[i],:] = np.array([(th[5*i],th[5*i+1],th[5*i+2],th[5*i+3],th[5*i+4])])
    trsf[Nm_ref,:] = np.array([(0.,0.,0.,1.,1.)])
    xp = th[starsdim]+1j*th[starsdim+1]
    xpm = th[starsdim+2]+1j*th[starsdim+3]
    xp_ref = coords_psr[:,0,0]+1j*coords_psr[:,0,1]
    pm_im = np.zeros([N_im],dtype=complex)
    for i in Nm_epc1:
        pm_im[i] = xpm*DeltaEpc1
    for i in Nm_epc2:
        pm_im[i] = xpm*DeltaEpc2
    probpsr = np.empty([N_im])
    for i in range(N_im):
        delxp = (exp(1j*trsf[i,2])*(xp+pm_im[i])+trsf[i,0]+1j*trsf[i,1])
        delxp = delxp.real*trsf[i,3]+1j*delxp.imag*trsf[i,4] - xp_ref[i]
        rotdelxp = delxp*exp(1j*coords_psr[i,0,4])
        probpsr[i] = (rotdelxp.real/coords_psr[i,0,2])**2 + (rotdelxp.imag/coords_psr[i,0,3])**2
    return lnprob(th,coords_ref,Nm_ref)-0.5*probpsr.sum()

p0=[]
for i in range(N_im-1):
    p0.append(0.)
    p0.append(0.)
    p0.append(0.)
    p0.append(1.)
    p0.append(1.)
for j in range(N_stars):
    p0.append(coords_ref[Nm_ref,j,0]*(0.995+1e-2*np.random.rand()))
    p0.append(coords_ref[Nm_ref,j,1]*(0.995+1e-2*np.random.rand()))
#if withpsr == "Y":
p0.append(coords_psr[Nm_ref,0,0]*(0.995+1e-2*np.random.rand()))
p0.append(coords_psr[Nm_ref,0,1]*(0.995+1e-2*np.random.rand()))
p0.append(0.)
p0.append(0.)

#print p0
#print len(p0)

#if withpsr == "Y":
ndim = 5*(N_im-1)+2*N_stars + 4
#else:
#    ndim     = 5*(N_im-1)+2*N_stars

pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

if firstrun == "N":
    inf   = open(LastPos)
    lines = inf.readlines()
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    inf.close()

#if withpsr == "Y":
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_psr, args=(), threads=10)
#else:
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(coords_ref,Nm_ref),threads=10)

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
