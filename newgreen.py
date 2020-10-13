#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import dustmaps
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from dustmaps.bayestar import BayestarQuery
import dustmaps.leike2020
#dustmaps.leike2020.fetch()
from dustmaps.leike2020 import Leike2020Query
from astropy import units

l = 266.2 * units.deg
b = -1.2 * units.deg

"""
ra = "8h52m1.38s"
dec = "-46d17m53.34s"
"""

ndist = 120
d = np.linspace(0.1,10.,ndist) * units.kpc

nsmpl = 5

#coords = SkyCoord(l, b, distance=d, frame='galactic')
coords = SkyCoord(l, b, distance=d, frame='galactic')

#bayestar = BayestarQuery(max_samples=nsmpl, version='bayestar2019')
leike = Leike2020Query()
#ebv = bayestar(coords, mode='samples')
#ebvmed = bayestar(coords, mode='median')
ebv = leike(coords)
#ebvmed = leike(coords)
#ebvbestfit = bayestar(coords, mode='bestfit')

f = open("velajr_leike.dat", "w")
"""
for i in range(ndist):
    print(float(d[i]/(1.*units.kpc)))
    f.write("%.15E "%(float(d[i]/(1.*units.kpc))))
    for j in range(nsmpl):
        print(ebv[i,j])
        f.write(" %.15E "%(ebv[i,j]))
    f.write("\n")
"""
f.close()

#for i in range(nsmpl):
#    plt.plot(d,ebv,color='gray')
plt.plot(d,ebv,color='black')
#plt.plot(d,ebvbestfit,color='red')

plt.xlabel(r'$\rm Distance \, [ \, kpc \, ]$',fontsize=14)
plt.ylabel(r'$E(B-V)$',fontsize=14)
plt.tick_params(labelsize=14)
plt.minorticks_on()

plt.savefig("velajr_leike"+".png")
#plt.show()
