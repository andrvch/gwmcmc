#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import *
import random
import time
from cudakde import *

Mns = 1.4
Rns = 13.
kb = 1.38E-16
kev = 1.6022E-9
gr = math.sqrt(1 - 2.952 * Mns / Rns)

nbins1D = 100
nbins2D = 200

#nsm = 500000
#samples = read_data_nsmpl(sys.argv[1],nsm)
samples = read_data(sys.argv[1])
print samples.shape
samples = samples[samples.shape[0]-3:samples.shape[0]-1,:]
print samples.shape
#samples = samples[:,np.where(samples[-1,:]<14000)[0]]
#print samples.shape

pipi = 3.14159265359

mumu = np.sqrt(samples[0]**2+samples[1]**2)/8.*1.E3
#anan = np.arctan(samples[1]/samples[0])*180./pipi

samples[0] = 98.43458788 - 180/pipi*np.arcsin(samples[0]/8.*60.*1.E3/3600.*pipi/180.)
samples[1] = 6.542318409 - 180/pipi*np.arcsin(samples[1]/8.*60.*1.E3/3600.*pipi/180.)

npars = len(samples)

qlevel = float(sys.argv[2]) # percent
#quont = [0.999,0.99,0.95,0.90]
#quont = [0.99,0.95,0.90,0.68,0.40]
quont = [0.99,0.95,0.68,0.40]
eqh_inter = np.empty([npars,3])

fig, ax = plt.subplots(ncols=npars, nrows=npars)
zizi = []

xii,yii = np.mgrid[-5.:5.:nbins2D*1j,-5.:5.:nbins2D*1j]
def gauss(x,y):
    return np.exp(-0.5*(x**2+y**2))
zii = gauss(xii,yii)
levi,ziin = comp_lev(zii.flatten(),quont)
from astropy.io import fits

sttime = time.time()
for j in range(npars):
    for i in range(npars):
        if i == j:
            xi,zi = kde_gauss_cuda1d(samples[i],nbins1D)
            zin,eqh_inter[i,:] = prc(xi,zi,0.01*qlevel)
            ax[i,j].plot(xi,zin,color='blue')
            xqu = [eqh_inter[i,0],eqh_inter[i,-1],eqh_inter[i,-1],eqh_inter[i,0]]
            yqu = [zin.min(),zin.min(),zin.max()+3*(zin.max()-zin.min()),zin.max()+3*(zin.max()-zin.min())]
            ax[i,i].fill(xqu,yqu,color='0.75')
            ax[i,i].plot([eqh_inter[i,1],eqh_inter[i,1]],[zin.min(),zin.max()+3*(zin.max()-zin.min())],'--',color='black',linewidth=1.5)
            zizi.append(zin)
        elif i > j:
            xi,yi,zi = kde_gauss_cuda2d(samples[j],samples[i],nbins2D)
            lev,zin = comp_lev(zi,quont)
            print lev
            #ax[i,j].contourf(xi,yi,zin.reshape(xi.shape), lev, alpha=.35, cmap=plt.cm.Greens)
            ax[i,j].contour(xi,yi,zin.reshape(xi.shape), lev, colors='blue', linewidth=.5)
            #datain  = np.array(zi.reshape(xi.shape))
            hdu     = fits.PrimaryHDU(data=zin.reshape(xi.shape))
            hduhdr  = hdu.header
            hduhdr.set('CTYPE1', 'RA---TAN')
            hduhdr.set('CRVAL1', '%5.7f'% (xi.min()))
            hduhdr.set('CUNIT1', 'deg ')
            hduhdr.set('CRPIX1', '%5.7f'% (1.0))
            hduhdr.set('CDELT1', '%5.7f'% ((xi.max()-xi.min())/nbins2D))
            hduhdr.set('CTYPE2', 'DEC---TAN')
            hduhdr.set('CRVAL2', '%5.7f'% (yi.min()))
            hduhdr.set('CUNIT2', 'deg ')
            hduhdr.set('CRPIX2', '%5.7f'% (1.0))
            hduhdr.set('CDELT2', '%5.7f'% ((yi.max()-yi.min())/nbins2D))
            hdu.writeto(sys.argv[1]+'.fits')
            #ax[i,j].contourf(xii,yii,ziin.reshape(xii.shape), lev, alpha=.35, cmap=plt.cm.Greens)
            #if i < npars-1:
            #    ax[i,j].contour(xii,yii,ziin.reshape(xii.shape), levi, colors='black', linewidth=.5)
        elif j > i:
            ax[i,j].set_visible(False)
print "gpu:"
print time.time()-sttime

for i in range(npars):
    ax[i,i].set_ylabel("p.d.f.")
    ax[i,i].yaxis.set_label_position("right")

ax[1,0].set_ylabel(r'$\mu_{\delta}$')
ax[1,0].set_xlabel(r'$\mu_{\alpha}\cos{\delta}$')

for i in range(1,npars):
    ax[i,i].yaxis.tick_right()
    setp([a.get_xticklabels() for a in ax[:npars-1,i]], visible=False)
    setp([a.get_yticklabels() for a in ax[i,1:i]], visible=False)

for i in range(npars):
    for j in range(npars):
        setp(ax[i,j].get_xticklabels(), rotation=0)

ax[0,0].yaxis.tick_right()
setp([a.get_xticklabels() for a in ax[:npars-1,0]], visible=False)

for i in range(npars):
    print eqh_inter[i,:]

for j in range(npars):
    for i in range(npars):
        ax[i,j].set_xlim(samples[j].min()-0.001*((samples[j].max()-samples[j].min())),samples[j].max()+0.001*((samples[j].max()-samples[j].min())))
        if i == j:
            ax[i,j].set_ylim(zizi[i].min()+0.001*(zizi[i].max()-zizi[i].min()),zizi[i].max()+0.05*(zizi[i].max()-zizi[i].min()))
        elif i > j:
            ax[i,j].set_ylim(samples[i].min()-0.05*(samples[i].max()-samples[i].min()), samples[i].max()+0.05*(samples[i].max()-samples[i].min()))

#plt.show()
plt.savefig(sys.argv[1]+"trnglANG4"+".jpg")
