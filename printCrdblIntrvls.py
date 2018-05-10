#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import random
import time

from cudakde import *

nbins1D = 100
nbins2D = 200

qqlevel = float(sys.argv[1])   # percent
quont = [0.99,0.90,0.68,0.40]

halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

pars = read_data(sys.argv[2])
npars = len(pars)

eqh_inter = np.empty([npars,len(quantiles)])

sttime=time.time()

for i in range(npars):
    xi,zi = kde_gauss_cuda1d(pars[i],nbins1D)
    zin,eqh_inter[i,:] = prc(xi,zi,qqq)
    print eqh_inter[i,:]

print "gpu:"
print time.time()-sttime
