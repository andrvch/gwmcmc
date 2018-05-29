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

pars = read_data(sys.argv[2])
npars = len(pars)

pars[0] = gr * kb * 10**pars[0] / kev
pars[1] = 10**pars[1]
pars[6] = 10**pars[6]

eqh_inter = np.empty([npars,len(quantiles)])

sttime=time.time()

for i in range(npars):
    xi,zi = kde_gauss_cuda1d(pars[i],nbins)
    zin,eqh_inter[i,:] = prc(xi,zi,qqq)
    print eqh_inter[i,:]

print "gpu:"
print time.time()-sttime
