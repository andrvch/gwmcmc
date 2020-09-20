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
from astropy.io import fits

pi = 3.14159265358979323846

fitsFls = [sys.argv[1],sys.argv[2],sys.argv[3]] # (files position_input or position_input_11)
regFls = [sys.argv[4],sys.argv[5],sys.argv[6]]

def compareRegAndFits(regFls,fitsFls):
    nim = len(regFls)
    regs = []
    for i in range(nim):
        regs.append(pyregion.open(regFls[i]))
    nst = shape(regs[0])[0]
    id = np.empty([nim,nst])
    for i in range(nim):
        for j in range(nst):
            id[i,j] = int(re.search(r'\d+',regs[i][j].comment)[0])
    npr = shape(regs[0][0].coord_list)[0]
    ell = np.empty([nim,nst,npr])
    for i in range(nim):
        for j in range(nst):
            for k in range(npr):
                ell[i,int(id[i,j]),k] = regs[i][j].coord_list[k]
    hduls = []
    for i in range(nim):
        hduls.append(fits.open(fitsFls[i]))
    vals = []
    for i in range(nim):
        data = hduls[i][1].data
        n2 = len(data)
        dat = np.empty([n2,4])
        for l in range(n2):
            dat[l,0] = data['X'][l]
            dat[l,1] = data['Y'][l]
            dat[l,2] = data['X_ERR'][l]
            dat[l,3] = data['Y_ERR'][l]
        vals.append(dat)
    coords = np.empty([nim,nst,npr])
    for i in range(nim):
        for j in range(nst):
            res = np.where(np.floor(vals[i][:,0])==math.floor(ell[i,j,0]))
            if len(res[0]) > 0:
                ell[i,j,0] = vals[i][res[0][0],0]
                ell[i,j,1] = vals[i][res[0][0],1]
                ell[i,j,2] = vals[i][res[0][0],2]
                ell[i,j,3] = vals[i][res[0][0],3]
    return ell

ell = compareRegAndFits(regFls,fitsFls)

N_im = shape(ell)[0]
N_stars = shape(ell)[1]
N_pars = shape(ell)[2]

for i in range(N_im):
    print("image number -- %i"%i)
    for j in range(N_stars):
        print(ell[i,j,0],ell[i,j,1],j)

xref = ell[:,:,0]+1j*ell[:,:,1]

ref_indx = [0,1,2,3,4,5]
n_ref = len(ref_indx)

print("Number of stars -- %i"%n_ref)
print("Number of images -- %i"%N_im)

motchErr = 0.07 # inlclude 0.07 systematic error

aell = np.sqrt(ell[:,:,2]**2+motchErr**2)
bell = np.sqrt(ell[:,:,3]**2+motchErr**2)

sampleName = sys.argv[7]
samples = read_data(sampleName)
print(samples.shape)

npars = np.shape(samples)[0]
N_samples = np.shape(samples)[1]

delx = np.zeros([N_im,n_ref,N_samples],dtype=complex)
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
    delx = np.zeros([N_im,n_ref],dtype=complex)
    for i in range(N_im):
        for k in range(n_ref):
            #if k == 3:
            #    delx[i,k] = pm[i,0]+1j*pm[i,1] + trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
            #else:
            delx[i,k] = trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xref[0,ref_indx[k]] - xref[i,ref_indx[k]]
    return delx

for i in range(N_samples):
    delx[:,:,i] = residuals(samples[:-1,i])

nbins = 100

qqlevel = float(sys.argv[8])   # percent
quont = [0.99,0.90,0.68,0.40]
halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

eqh_inter = np.empty([N_im,n_ref,2,len(quantiles)])

print("delta x:")
for i in range(1,N_im):
    for j in range(n_ref):
        print("image number -- %i"%i)
        print("star number -- %i"%j)
        xi,zi = kde_gauss_cuda1d(delx[i,j,:].real,nbins)
        zin,eqh_inter[i,j,0,:] = prc(xi,zi,qqq)
        print(eqh_inter[i,j,0,:])

print("delta y:")
for i in range(1,N_im):
    for j in range(n_ref):
        print("image number -- %i"%i)
        print("star number -- %i"%j)
        xi,zi = kde_gauss_cuda1d(delx[i,j,:].imag,nbins)
        zin,eqh_inter[i,j,1,:] = prc(xi,zi,qqq)
        print(eqh_inter[i,j,1,:])

starnumber = linspace(0,n_ref-1,num=n_ref)

fig, ax = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.1)

colors = ['g','y','b']
labels = ['11123','19165','20876']

for i in range(1,N_im):
    ax[0].errorbar(starnumber,eqh_inter[i,:,0,1], yerr=[eqh_inter[i,:,0,1]-eqh_inter[i,:,0,0],eqh_inter[i,:,0,2]-eqh_inter[i,:,0,1]],color=colors[i],fmt='*',capsize=10,label=labels[i])
    ax[1].errorbar(starnumber,eqh_inter[i,:,1,1], yerr=[eqh_inter[i,:,1,1]-eqh_inter[i,:,1,0],eqh_inter[i,:,1,2]-eqh_inter[i,:,1,1]],color=colors[i],fmt='*',capsize=10,label=labels[i])

ax[0].legend()
#ax[1].legend()

ax[0].errorbar([0,n_ref-1],[0,0],color='k',fmt='--')
ax[1].errorbar([0,n_ref-1],[0,0],color='k',fmt='--')
#ax[1].set_xticks(np.arange(0, 5, step=1))
#ax[0].set_xticks(np.arange(0, 5, step=1))
#ax[1].set_xlim(0.7,4.3)
#ax[0].set_xlim(0.7,4.3)
ax[0].set_ylabel("Delta X")
ax[1].set_ylabel("Delta Y")

plt.savefig(sampleName+"PlainRes"+"%2.1f"%(qqlevel)+".png")
