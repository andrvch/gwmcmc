#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pylab import *
from astropy.io import ascii
import numpy as np
import emcee
import scipy.optimize as op
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

pi = 3.14159265358979323846

Crds_file = sys.argv[1] # (files position_input or position_input_11)
NameChain = sys.argv[2]
firstrun = sys.argv[3]
thread_num = float(sys.argv[4])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"%(thread_num)+".dat"
LastPos = NameChain+"_"+"LastPos"+".dat"

nsteps = 5000
nwalkers = 256
print("Number of walkers -- %i"%nwalkers)
print("Number of steps -- %i"%nsteps)

def parse_input(in_file):
    f = open(in_file)
    lines = f.readlines()
    n = len(lines)
    coords = []
    for i in range(n):
        coords.append([float(x) for x in lines[i].split()])
    coords_1 = np.empty([len(coords),6])
    for i in range(len(coords)):
        coords_1[i] = np.array([coords[i]])
    return coords_1

coords = parse_input(Crds_file)
coords_1 = np.array([coords[:4],coords[4:8],coords[8:12]])
N_im = len(coords_1)
N_stars = len(coords_1[0])
print("Number of stars -- %i"%N_stars)
print("Number of images -- %i"%N_im)

cosdelta = cos(pi/180.*coords_1[0,3,2])
raOff = (coords_1[:,:,1]-coords_1[0,3,1])*3600*cosdelta
decOff = (coords_1[:,:,2]-coords_1[0,3,2])*3600
errraOff = coords_1[:,:,3]*3600*cosdelta
errdecOff = coords_1[:,:,4]*3600
motchErr = 0.07 # inlclude 0.07 systematic error
errraOff = np.sqrt(errraOff**2 + motchErr**2)
errdecOff = np.sqrt(errdecOff**2 + motchErr**2)
xref = raOff+1j*decOff

print("Offsets, ra:")
print(raOff)
print(decOff)
print("Offset errors, ra:")
print(errraOff)
print(errdecOff)

def lnprior(th):
    if th[-2] >= 0. and th[-1] <= 2*pi and th[-1] >= 0.:
        return 0.0
    return -np.inf

def lnprob(th):
    lp = lnprior(th)
    if not np.isfinite(lp):
        return -np.inf
    trsf = np.empty([N_im,3])
    trsf[0,:] = np.array([(0.,0.,0.)])
    for i in range(1,N_im):
        trsf[i,:] = np.array([(th[3*(i-1)],th[3*(i-1)+1],th[3*(i-1)+2])])
    xs1 = np.empty([N_stars,2])
    for i in range(N_stars):
        xs1[i,0] = th[3*(N_im-1)+2*i]
        xs1[i,1] = th[3*(N_im-1)+2*i+1]
    xs = xs1[:,0]+1j*xs1[:,1]
    pm = np.empty([N_im,2])
    pm[0,:] = np.array([(0.,0.)])
    for i in range(1,N_im):
        pm[i,:] = np.array([(th[-2],th[-1])])
    prob = np.empty([N_im,N_stars])
    for i in range(N_im):
        for k in range(N_stars):
            if k == 3:
                delx = pm[i,0]*exp(1j*pm[i,1]) + trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
            else:
                delx = trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
            prob[i,k] = (delx.real/errraOff[i,k])**2 + (delx.imag/errdecOff[i,k])**2
    return -0.5*prob.sum()

p0=[]
for i in range(N_im-1):
    p0.append(0.)
    p0.append(0.)
    p0.append(0.)
for j in range(N_stars):
    p0.append(raOff[0,j])
    p0.append(decOff[0,j])
p0.append(0.1)
p0.append(0.1)
print("Starting parameters vector --")
print(p0)

ndim = 3*(N_im-1) + 2*N_stars + 2
print("Number of parameters -- %i"%ndim)
pos = [p0 + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]

if firstrun == "N":
    inf = open(LastPos)
    lines = inf.readlines()
    nsmpl = len(lines)
    prepos = []
    for i in range(nsmpl):
        prepos.append([float(x) for x in lines[i].split()])
    initial = prepos
    inf.close()
elif firstrun == "Y":
    initial = pos

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    #print("{0:.1f} times faster than serial".format(serial_time / multi_time))

samples = sampler.get_chain()

lastpos = samples[-1,:,:]
f = open(LastPos, "w")
n1, n2 = shape(lastpos)
for i in range(n1):
    for j in range(n2):
        f.write("%.15E "%(lastpos[i,j]))
    f.write("\n")
f.close()

nPlot = ndim

steps = linspace(1,nsteps,num=nsteps)
fig, ax = plt.subplots(nrows=nPlot)
plt.subplots_adjust(hspace=0.1)

for i in range(nPlot):
    for j in range(nwalkers):
        #print(np.reshape(samples[:,j,i],(nsteps)))
        ax[i].errorbar(steps,samples[:,j,i])

#setp([a.get_xticklabels() for a in ax[:nPlot-1]], visible=False)
#for a in ax[:nPlot-1]:
#    for tick in a.get_xticklabels():
#        tick

#plt.show()
plt.savefig(sys.argv[2]+"chain"+".png")
burn = 0 #int(raw_input("How many steps to discard: "))

outwalk = samples[burn:,:,:].reshape((-1,ndim))
outlike = sampler.get_log_prob()[burn:,:].reshape((-1))

f = open(SampleName,"w")
n1, n2 = shape(outwalk)
for i in range(n1):
    for j in range(n2):
        f.write(" %.15E "%(outwalk[i,j]))
    f.write(" %.15E "%(-2.*outlike[i] ))
    f.write("\n")
f.close()
