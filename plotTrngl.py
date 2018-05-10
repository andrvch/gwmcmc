#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *
import random
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycuda import driver, compiler, gpuarray, tools
import time
# -- initialize the device
import pycuda.autoinit
from cudakde import *

nbins1D   = 100
nbins2D   = 200

qqlevel   = float(sys.argv[1])   # percent
quont     = [0.99,0.90,0.68,0.40]

halfqq    = (100 - qqlevel)*0.5
qqq       = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

pars = read_data(sys.argv[2])
npars = len(pars)

fig, ax = plt.subplots(ncols=npars, nrows=npars)

eqh_inter = np.empty([npars,len(quantiles)])
zizi = []

sttime=time.time()
for j in range(npars):
    for i in range(npars):
        if i == j:
            xi,zi = kde_gauss_cuda1d(pars[i],nbins1D)
            zin,eqh_inter[i,:] = prc(xi,zi,qqq)
            ax[i,j].plot(xi,zin,color='blue')
            xqu = [eqh_inter[i,0],eqh_inter[i,-1],eqh_inter[i,-1],eqh_inter[i,0]]
            yqu = [zin.min(),zin.min(),zin.max()+3*(zin.max()-zin.min()),zin.max()+3*(zin.max()-zin.min())]
            ax[i,i].fill(xqu,yqu,color='0.75')
            ax[i,i].plot([eqh_inter[i,1],eqh_inter[i,1]],[zin.min(),zin.max()+3*(zin.max()-zin.min())],'--',color='black',linewidth=1.5)                
            zizi.append(zin)
        elif i > j:
            xi,yi,zi = kde_gauss_cuda2d(pars[j],pars[i],nbins2D)
            lev,zin  = comp_lev(zi,quont)
            ax[i,j].contourf(xi,yi,zin.reshape(xi.shape), lev, alpha=.35, cmap=plt.cm.Greens)
            ax[i,j].contour(xi,yi,zin.reshape(xi.shape), lev, colors='black', linewidth=.5)
        elif j > i:
            ax[i,j].set_visible(False)
print "gpu:"
print time.time()-sttime

for i in range(npars):
    ax[i,i].set_ylabel(r'$\rm p.d.f.$',fontsize=18)
    ax[i,i].yaxis.set_label_position("right")
    
for i in range(1,npars):
    ax[i,i].yaxis.tick_right()
    setp([a.get_xticklabels() for a in ax[:npars-1,i]], visible=False)
    setp([a.get_yticklabels() for a in ax[i,1:i]], visible=False)

for i in range(npars):
    for j in range(npars):
        setp(ax[i,j].get_xticklabels(), rotation=45)

ax[0,0].yaxis.tick_right()
setp([a.get_xticklabels() for a in ax[:npars-1,0]], visible=False)  

for i in range(npars):
    print eqh_inter[i,:]

for j in range(npars):
    for i in range(npars):
        ax[i,j].set_xlim(pars[j].min()-0.001*((pars[j].max()-pars[j].min())),pars[j].max()+0.001*((pars[j].max()-pars[j].min())))
        if i == j:
            ax[i,j].set_ylim(zizi[i].min()+0.001*(zizi[i].max()-zizi[i].min()),zizi[i].max()+0.05*(zizi[i].max()-zizi[i].min()))
        elif i > j:
            ax[i,j].set_ylim(pars[i].min()-0.05*(pars[i].max()-pars[i].min()), pars[i].max()+0.05*(pars[i].max()-pars[i].min()))

plt.show()
