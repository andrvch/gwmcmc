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
#samples = samples[:samples.shape[0],:]

npars = len(samples)

qlevel = float(sys.argv[2]) # percent
#quont = [0.999,0.99,0.95,0.90]
#quont = [0.99,0.95,0.90,0.68,0.40]
quont = [0.999,0.99,0.90,0.68,0.40]
eqh_inter = np.empty([npars,3])

zizi = []

xii,yii = np.mgrid[-5.:5.:nbins2D*1j,-5.:5.:nbins2D*1j]
def gauss(x,y):
    return np.exp(-0.5*(x**2+y**2))
zii = gauss(xii,yii)
levi,ziin = comp_lev(zii.flatten(),quont)

fig, ax = plt.subplots()

xi,zi = kde_gauss_cuda1d(samples[0],nbins1D)
zin,eqh_inter[0,:] = prc(xi,zi,0.01*qlevel)
ax.plot(xi,zin,color='blue')
xqu = [eqh_inter[0,0],eqh_inter[0,-1],eqh_inter[0,-1],eqh_inter[0,0]]
yqu = [zin.min(),zin.min(),zin.max()+0.16*(zin.max()-zin.min()),zin.max()+0.16*(zin.max()-zin.min())]
ax.fill(xqu,yqu,color='0.75')
ax.plot([eqh_inter[0,1],eqh_inter[0,1]],[zin.min(),zin.max()+3*(zin.max()-zin.min())],'--',color='black',linewidth=1.5)

ax.set_xlim(592.420,592.426)
ax.set_ylim(0.0,zin.max()+0.16*(zin.max()-zin.min()))

ax.set_xlabel("frequency",fontsize=11)
ax.set_ylabel("p.d.f.",fontsize=11)
plt.setp(ax.get_yticklabels(), fontsize=11)
plt.setp(ax.get_xticklabels(), fontsize=11)

#plt.show()
plt.savefig(sys.argv[1]+"trnglFreq"+".jpg")
