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
PIPI = 3.14159265359
pc = 3.08567802e18

edot = 1.2e35

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

samples[0] = samples[0] #10**samples[0]*kb/1.6022E-12/redshift
samples[1] = samples[1] + log10(1.E5*Rns)
samples[3] = samples[3]
samples[5] = samples[5]
samples[6] = samples[6]*10.
samples[7] = samples[7] + log10(pc)

samples2 = np.copy(samples)

def plflux(g,k,e1,e2):
    if g != 2:
        f = (e2**(-g+2)-e1**(-g+2))/(-g+2)
    else:
        f = log(e2)-log(e1)
    return log10(kev) + k + log10(f)

samples2[0] = log10(4.*PIPI*5.6704e-5) + 2.*samples[1] + 4.*samples[0]
samples2[1] = samples2[0] - log10(4.*PIPI) - 2.*samples[7]

for i in range(shape(samples)[1]):
    samples2[2,i] = plflux(samples[2,i],samples[3,i],2.,10.)
    samples2[3,i] = plflux(samples[4,i],samples[5,i],2.,10.)

samples2[4] = log10(4.*PIPI) + 2.*samples[7] + samples2[2]
samples2[5] = log10(4.*PIPI) + 2.*samples[7] + samples2[3]
samples2[6] = samples2[4] - log10(edot)
samples2[7] = samples2[5] - log10(edot)

npars = 8

eqh_inter = np.empty([npars,len(quantiles)])

sttime=time.time()

for i in range(npars):
    xi,zi = kde_gauss_cuda1d(samples2[i],nbins)
    zin,eqh_inter[i,:] = prc(xi,zi,qqq)
    print eqh_inter[i,:]

f = open(sys.argv[1]+"."+"%2.2f"%(qqlevel)+"."+"flux.crdbl", "w")
for i in range(npars):
    for j in range(3):
        f.write(" %.15E "%(eqh_inter[i,j]))
    f.write("\n")
f.close()

print "gpu:"
print time.time()-sttime
