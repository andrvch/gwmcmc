#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import time
from cudakde import *

qqlevel = 90  # percent
quont = [0.99,0.90,0.68,0.40]
halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

npars = 10
kde = readspectra(npars,sys.argv[1])

eqh_inter = np.empty([npars,len(quantiles)])

for i in range(npars):
    #xi,zi = kde_gauss_cuda1d(samples[i],nbins)
    zin,eqh_inter[i,:] = prc(kde[i][0],kde[i][1],qqq)
    print eqh_inter[i,:]

f = open(sys.argv[1]+"."+"%2.2f"%(qqlevel)+"."+"crdbl", "w")
for i in range(npars):
    for j in range(3):
        f.write(" %.15E "%(eqh_inter[i,j]))
    f.write("\n")
f.close()
