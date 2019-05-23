#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from cudakde import *

Mns = 1.4
Rns = 13.
kb = 1.38E-16
kev = 1.6022E-9
gr = math.sqrt(1 - 2.952 * Mns / Rns)
redshift = 1. / gr

qqlevel = 90  # percent
quont = [0.99,0.90,0.68,0.40]
halfqq = (100 - qqlevel)*0.5
qqq = 0.01*qqlevel
quantiles = [halfqq,50,qqlevel+halfqq]

npars = 10
kde = readspectra(npars,sys.argv[1]+"kde"+".kde")

kde[0][0] = 10**kde[0][0]*kb/1.6022E-12/redshift
kde[1][0] = 10**kde[1][0]*Rns
kde[3][0] = 10**kde[3][0]/1.E-5
kde[5][0] = 10**kde[5][0]/1.E-5
kde[6][0] = kde[6][0]*10.
kde[7][0] = 10**kde[7][0]/1.E3

#kde[0][0] = 10**kde[0][0]*1.E3 #*kb/1.6022E-12/redshift
#kde[1][0] = 10**(0.5*kde[1][0]+kde[7][0])
#kde[3][0] = 10**kde[3][0]/1.E-5
#kde[5][0] = 10**kde[5][0]/1.E-5
#kde[6][0] = kde[6][0]*10.
#kde[7][0] = 10**kde[7][0]/1.E3

eqh_inter = np.empty([npars,len(quantiles)])

for i in range(npars):
    zin,eqh_inter[i,:] = prc(kde[i][0],kde[i][1],qqq)
    print eqh_inter[i,:]

f = open(sys.argv[1]+"_"+"%2.2f"%(qqlevel)+"."+"crdbl", "w")
for i in range(npars):
    for j in range(3):
        f.write(" %.15E "%(eqh_inter[i,j]))
    f.write("\n")
f.close()
