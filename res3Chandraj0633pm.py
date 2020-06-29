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
print samples.shape

npars = np.shape(samples)[0]
N_samples = np.shape(samples)[1]

delx = np.empty([N_im,N_stars,])

def residuals(th):
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
    delx = np.empty([N_im,N_stars])
    for i in range(N_im):
        for k in range(N_stars):
            if k == 3:
                delx[i,k] = pm[i,0]*exp(1j*pm[i,1]) + trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
            else:
                delx[i,k] = trsf[i,0]+1j*trsf[i,1] + exp(1j*trsf[i,2])*xs[k] - xref[i,k]
    return delx

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
plt.savefig(NameChain+"res"+".png")
