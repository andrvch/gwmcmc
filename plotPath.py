#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
#matplotlib.use('Agg')
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
samples = samples[:samples.shape[0]-1,:]
#samples = samples[:3,:]
print samples.shape
#samples = samples[:,np.where(samples[-1,:]<14000)[0]]
#print samples.shape

xx = np.linspace(0,1,100)

for i in range(5):
    plt.plot(xx,samples[:,i])

plt.show()
#plt.savefig(sys.argv[1]+".trngl"+".jpg")
