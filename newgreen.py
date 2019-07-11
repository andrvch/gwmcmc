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

l = 205.09 * units.deg
b = -0.93 * units.deg
ndist = 120
d = np.linspace(0.1,10.,ndist) * units.kpc
nsmpl = 5

coords = SkyCoord(l, b, distance=d, frame='galactic')

bayestar = BayestarQuery(max_samples=nsmpl, version='bayestar2019')

ebv = bayestar(coords, mode='samples')
ebvmed = bayestar(coords, mode='median')
#ebvbestfit = bayestar(coords, mode='bestfit')

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
#plt.plot(d,ebvbestfit,color='red')

plt.savefig("Green19samples"+".jpg")
#plt.show()
