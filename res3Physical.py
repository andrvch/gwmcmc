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
#import emcee
from cudakde import *
#import scipy.optimize as op
#from multiprocessing import Pool
#os.environ["OMP_NUM_THREADS"] = "1"
import pyregion
import re

pi = 3.14159265358979323846

reg_files = [sys.argv[1],sys.argv[2],sys.argv[3]]

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
        ell[i,int(star_ids[i,j]),N_pars] =  int(star_ids[i,j])

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


samples = read_data(sys.argv[4])
print(samples.shape)

npars = np.shape(samples)[0]
N_samples = np.shape(samples)[1]

delx = np.zeros([N_im,N_stars,N_samples],dtype=complex)
#exit()

def residuals(th):
    trsf = np.empty([N_im,3])
    trsf[0,:] = np.array([(0.,0.,0.)])
    for i in range(1,N_im):
        trsf[i,:] = np.array([(th[3*(i-1)],th[3*(i-1)+1],th[3*(i-1)+2])])
    """
    xs1 = np.empty([N_stars,2])
    for i in range(N_stars):
        xs1[i,0] = th[3*(N_im-1)+2*i]
        xs1[i,1] = th[3*(N_im-1)+2*i+1]
    xs = xs1[:,0]+1j*xs1[:,1]
    pm = np.empty([N_im,2])
    pm[0,:] = np.array([(0.,0.)])
    for i in range(1,N_im):
        pm[i,:] = np.array([(0.,0.)]) #np.array([(th[-2],th[-1])])
    """
    delx = np.zeros([N_im,N_stars],dtype=complex)
    for i in range(N_im):
        for k in range(N_stars):
            #if k == 3:
            #    delx[i,k] = pm[i,0]+1j*pm[i,1] + trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
            #else:
            delx[i,k] = trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xref[0,k] - xref[i,k]
    return delx

for i in range(N_samples):
    delx[:,:,i] = residuals(samples[:-1,i])

nbins = 100

qqlevel = float(sys.argv[5])   # percent
quont = [0.99,0.90,0.68,0.40]
halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

eqh_inter = np.empty([N_im,N_stars,2,len(quantiles)])

print("delta x:")
for i in range(1,N_im):
    for j in range(N_stars):
        print("image number -- %i"%i)
        print("star number -- %i"%j)
        xi,zi = kde_gauss_cuda1d(delx[i,j,:].real,nbins)
        zin,eqh_inter[i,j,0,:] = prc(xi,zi,qqq)
        print(eqh_inter[i,j,0,:])

print("delta y:")
for i in range(1,N_im):
    for j in range(N_stars):
        print("image number -- %i"%i)
        print("star number -- %i"%j)
        xi,zi = kde_gauss_cuda1d(delx[i,j,:].imag,nbins)
        zin,eqh_inter[i,j,1,:] = prc(xi,zi,qqq)
        print(eqh_inter[i,j,1,:])

starnumber = linspace(0,N_stars,num=N_stars)

fig, ax = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.1)

colors = ['g','y','b']
labels = ['11123','19165','20876']

for i in range(1,N_im):
    ax[0].errorbar(starnumber,eqh_inter[i,:,0,1], yerr=[eqh_inter[i,:,0,1]-eqh_inter[i,:,0,0],eqh_inter[i,:,0,2]-eqh_inter[i,:,0,1]],color=colors[i],fmt='*',capsize=10,label=labels[i])
    ax[1].errorbar(starnumber,eqh_inter[i,:,1,1], yerr=[eqh_inter[i,:,1,1]-eqh_inter[i,:,1,0],eqh_inter[i,:,1,2]-eqh_inter[i,:,1,1]],color=colors[i],fmt='*',capsize=10,label=labels[i])

ax[0].legend()
#ax[1].legend()

ax[0].errorbar([0,N_stars],[0,0],color='k',fmt='--')
ax[1].errorbar([0,N_stars],[0,0],color='k',fmt='--')
#ax[1].set_xticks(np.arange(0, 5, step=1))
#ax[0].set_xticks(np.arange(0, 5, step=1))
#ax[1].set_xlim(0.7,4.3)
#ax[0].set_xlim(0.7,4.3)
ax[0].set_ylabel("Delta X")
ax[1].set_ylabel("Delta Y")

plt.savefig(sys.argv[4]+"PlainRes"+"%2.1f"%(float(sys.argv[5]))+".png")
