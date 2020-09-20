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
import pyregion
import re

os.environ["OMP_NUM_THREADS"] = "1"

pi = 3.14159265358979323846

nsteps = 1024
nwalkers = 256

print("Number of walkers -- %i"%nwalkers)
print("Number of steps -- %i"%nsteps)

reg_files = [sys.argv[1],sys.argv[2],sys.argv[3]] # input reg files

N_im = len(reg_files)

regs = []

for i in range(N_im):
    regs.append(pyregion.open(reg_files[i]))

N_stars = shape(regs[0])[0]

star_ids = np.empty([N_im,N_stars])

for i in range(N_im):
    for j in range(N_stars):
        star_ids[i,j] = int(re.search(r'\d+',regs[i][j].comment)[0])

N_pars = shape(regs[0][0].coord_list)[0]

ell = np.empty([N_im,N_stars,N_pars+1])

for i in range(N_im):
    for j in range(N_stars):
        for k in range(N_pars):
            ell[i,int(star_ids[i,j]),k] = regs[i][j].coord_list[k]
        ell[i,int(star_ids[i,j]),N_pars] = int(star_ids[i,j])

for i in range(N_im):
    print("image number -- %i"%i)
    for j in range(N_stars):
        print(ell[i,j,0],ell[i,j,1],ell[i,j,N_pars])

xref = ell[:,:,0]+1j*ell[:,:,1]

print("Number of stars -- %i"%N_stars)
print("Number of images -- %i"%N_im)

motchErr = 0.0 # inlclude 0.07 systematic error

aell = np.sqrt(ell[:,:,2]**2+motchErr**2)
bell = np.sqrt(ell[:,:,3]**2+motchErr**2)

def lnprob(th):
    trsf = np.empty([N_im,3])
    trsf[0,:] = np.array([(0.,0.,0.)])
    for i in range(1,N_im):
        trsf[i,:] = np.array([(th[3*(i-1)],th[3*(i-1)+1],th[3*(i-1)+2])])
    xs1 = np.empty([N_stars-1,2])
    for i in range(N_stars-1):
        xs1[i,0] = th[3*(N_im-1)+2*i]
        xs1[i,1] = th[3*(N_im-1)+2*i+1]
    xs = xs1[:,0]+1j*xs1[:,1]
    '''
    pm = np.empty([N_im,2])
    pm[0,:] = np.array([(0.,0.)])
    for i in range(1,N_im):
        pm[i,:] = np.array([(0.,0.)]) #np.array([(th[-2],th[-1])])
    '''
    prob = np.empty([N_im,N_stars-1])
    for i in range(N_im):
        for k in range(1,N_stars):
            '''
            if k == 3:
                delx = pm[i,0]*exp(1j*pm[i,1]) + trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
            else:
            '''
            delx = trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k-1] - xref[i,k]
            delx = delx*exp(1j*pi*ell[i,k,4]/180.)
            prob[i,k-1] = (delx.real/aell[i,k])**2 + (delx.imag/bell[i,k])**2
    return -0.5*prob.sum()

p0=[]
for i in range(N_im-1):
    p0.append(0.)
    p0.append(0.)
    p0.append(0.)
for i in range(1,N_stars):
    p0.append(ell[0,i,0])
    p0.append(ell[0,i,1])
#p0.append(0.1)
#p0.append(0.1)
print("Starting parameters vector --")
print(p0)

ndim = 3*(N_im-1) + 2*(N_stars-1) # + 2
print("Number of parameters -- %i"%ndim)
print("Number of degrees of freedom -- %i"%(2*(N_stars-1)*N_im-ndim))
pos = [p0 + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]

NameChain = reg_files[0]

firstrun = sys.argv[4]
thread_num = float(sys.argv[5])

SampleName = NameChain+"_"+"Thread"+"_"+"%1i"%(thread_num)+".dat"
LastPos = NameChain+"_"+"LastPos"+".dat"


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
