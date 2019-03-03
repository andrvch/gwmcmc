#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from cudakde import *

samples = read_data(sys.argv[1])
print samples.shape
#samples = samples[:samples.shape[0],:]
samples = samples[np.r_[0:3, samples.shape[0]-1],:]
print samples.shape
#samples = samples[:,np.where(samples[-1,:]<14000)[0]]
#print samples.shape

nwlkrs = int(sys.argv[2])
nprmtrs = shape(samples)[0]
nstps = int(shape(samples)[1]/float(nwlkrs))

print nstps, nprmtrs

wlkrs = np.empty([nprmtrs,nwlkrs,nstps])

for i in range(nstps):
    for j in range(nwlkrs):
        for k in range(nprmtrs):
            wlkrs[k,j,i] = samples[k,j+nwlkrs*i]

stps = linspace(1,nstps,num=nstps)
fig, ax = plt.subplots(nrows=nprmtrs)
plt.subplots_adjust(hspace=0.1)

for i in range(nprmtrs):
    for j in range(nwlkrs):
        ax[i].errorbar(stps,wlkrs[i,j,:])

#plt.show()
plt.savefig(sys.argv[1]+".jpg")
