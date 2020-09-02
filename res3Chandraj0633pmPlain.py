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

pi = 3.14159265358979323846

Crds_file = sys.argv[1]
NameChain = sys.argv[2]

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

samples = read_data(NameChain)
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

qqlevel = float(sys.argv[3])   # percent
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

starnumber = linspace(1,N_stars,num=N_stars)

fig, ax = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.1)

colors = ['g','y','b']
labels = ['11123','19165','20876']

for i in range(1,N_im):
    ax[0].errorbar(starnumber,eqh_inter[i,:,0,1], yerr=[eqh_inter[i,:,0,1]-eqh_inter[i,:,0,0],eqh_inter[i,:,0,2]-eqh_inter[i,:,0,1]],color=colors[i],fmt='*',capsize=10,label=labels[i])
    ax[1].errorbar(starnumber,eqh_inter[i,:,1,1], yerr=[eqh_inter[i,:,1,1]-eqh_inter[i,:,1,0],eqh_inter[i,:,1,2]-eqh_inter[i,:,1,1]],color=colors[i],fmt='*',capsize=10,label=labels[i])

ax[0].legend()
#ax[1].legend()

ax[0].errorbar([0,5],[0,0],color='k',fmt='--')
ax[1].errorbar([0,5],[0,0],color='k',fmt='--')
ax[1].set_xticks(np.arange(0, 5, step=1))
ax[0].set_xticks(np.arange(0, 5, step=1))
ax[1].set_xlim(0.7,4.3)
ax[0].set_xlim(0.7,4.3)
ax[0].set_ylabel("Delta RA")
ax[1].set_ylabel("Delta Dec")

plt.savefig(NameChain+"PlainRes"+"%2.1f"%(float(sys.argv[3]))+".png")
