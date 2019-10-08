#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import dustmaps
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery
from astropy import units
from cudakde import *

pars = read_data(sys.argv[1])
pars1 = read_data(sys.argv[2])

l = 205.09 * units.deg
b = -0.93 * units.deg
ndist = 124
dd = np.linspace(0.63,10.,ndist)
d = np.linspace(0.63,10.,ndist) * units.kpc
nsmpl = 5
coeff = 0.9014 # E(B-V) = coeff*E(g-r), see argonaut.skymaps.info/usage

coords = SkyCoord(l, b, distance=d, frame='galactic')
bayestar = BayestarQuery(max_samples=nsmpl, version='bayestar2019')

ebv = coeff*bayestar(coords, mode='samples')
ebvmed = coeff*bayestar(coords, mode='median')

f = open("Green19.dat", "w")

for i in range(ndist):
    print float(d[i]/(1.*units.kpc))
    f.write("%.15E "%(float(d[i]/(1.*units.kpc))))
    for j in range(nsmpl):
        print ebv[i,j]
        f.write(" %.15E "%(ebv[i,j]))
    f.write("\n")
f.close()

for i in range(nsmpl):
    plt.plot(d,ebv,color='gray')
plt.plot(d,ebvmed,color='black')

xqu = [dd.min(),dd.max(),dd.max(),dd.min()]
yqu = [pars[0,6]/7.,pars[0,6]/7.,pars[2,6]/7.,pars[2,6]/7.]
yqu1 = [pars1[0,6]/7.,pars1[0,6]/7.,pars1[2,6]/7.,pars1[2,6]/7.]
plt.fill(xqu,yqu,color='0.35',edgecolor='black',alpha=0.3)
#plt.fill(xqu,yqu1,color='0.75',edgecolor='blue',alpha=0.3)
plt.plot([dd.min(),dd.max()],[pars[1,6]/7.,pars[1,6]/7.],'--',color='black',zorder=10)
plt.plot([dd.min(),dd.max()],[pars1[1,6]/7.,pars1[1,6]/7.],'--',color='blue',zorder=11)
#plt.plot(d,ebvbestfit,color='red')

plt.xlabel(r'$\rm Distance \, [ \, kpc \, ]$',fontsize=14)
plt.ylabel(r'$E(B-V)$',fontsize=14)
plt.tick_params(labelsize=14)
plt.minorticks_on()

plt.savefig("Green19samples"+".eps")
#plt.show()
