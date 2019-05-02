#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import random
import time
from cudakde import *

Mns = 1.4
Rns = 13.
kb = 1.38E-16
kev = 1.6022E-9
gr = math.sqrt(1 - 2.952 * Mns / Rns)
redshift = 1. / gr

nbins = 100

qqlevel = 90  # percent
quont = [0.99,0.90,0.68,0.40]
halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

samples = read_data(sys.argv[1])
print samples.shape
#samples = samples[np.r_[0:8,14:samples.shape[0]-1],:]
samples = samples[np.r_[0:samples.shape[0]-1],:]
print samples.shape

npars = len(samples)

samples[0] = 10**samples[0] #*kb/1.6022E-12/redshift
samples[1] = 10**(0.5*samples[1])*10**samples[7]
samples[3] = 10**samples[3]/1.E-5
samples[5] = 10**samples[5]/1.E-5
samples[6] = samples[6]*10.
samples[7] = 10**samples[7]/1.E3

eqh_inter = np.empty([npars,len(quantiles)])

sttime=time.time()

for i in range(npars):
    xi,zi = kde_gauss_cuda1d(samples[i],nbins)
    zin,eqh_inter[i,:] = prc(xi,zi,qqq)
    print eqh_inter[i,:]

f = open(sys.argv[1]+"."+"%2.2f"%(qqlevel)+"."+"crdbl", "w")
for i in range(npars):
    for j in range(3):
        f.write(" %.15E "%(eqh_inter[i,j]))
    f.write("\n")
f.close()

print "gpu:"
print time.time()-sttime
