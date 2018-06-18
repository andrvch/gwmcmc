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

nbins = 100

qqlevel = float(sys.argv[1])   # percent
quont = [0.99,0.90,0.68,0.40]
halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

#nsm = 500000
#samples = read_data_nsmpl(sys.argv[2],nsm)
samples = read_data(sys.argv[2])
print samples.shape
samples = samples[np.r_[0:samples.shape[0]],:]
#print samples.shape
#samples = samples[:,np.where(samples[-1,:]<14000)[0]]
print samples.shape

npars = len(samples)

#samples[0] = gr * kb * 10**samples[0] / kev
norm = np.copy(samples[1])
dist = np.copy(samples[2])
radi = np.copy(samples[1])
#radi = dist + norm # + math.log10(Rns)
#samples[12] = radi

#samples[1] = samples[1] + samples[2]
#pars[1] = 10**pars[1]
#pars[3] = 10**pars[3]

#for i in range(samples.shape[1]):
#    print norm[i], dist[i], norm[i]+dist[i], radi[i]

eqh_inter = np.empty([npars,len(quantiles)])

sttime=time.time()

for i in range(npars):
    xi,zi = kde_gauss_cuda1d(samples[i],nbins)
    zin,eqh_inter[i,:] = prc(xi,zi,qqq)
    print eqh_inter[i,:]

f = open(sys.argv[2]+"."+"%2.0f"%(float(sys.argv[1]))+"."+"credible", "w")
for i in range(npars):
    for j in range(3):
        f.write(" %.15E "%(eqh_inter[i,j]))
    f.write("\n")
f.close()

print "gpu:"
print time.time()-sttime
