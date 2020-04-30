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

n = 4
m = 20
pipi = 3.14159265359

xxRad = 0.05
xxAng = np.empty([m])
xxCen = np.empty([2,n])
xx = np.empty([2,m,n])

for i in range(m):
    xxAng[i] = pipi/10.*i
#for j in range(n):
ii = np.array([0, 0, 1, 1])
jj = np.array([0, 1, 0, 1])
for j in range(n):
    xxCen[0,j] = 1./4. + 1./2.*ii[j]
    xxCen[1,j] = 1./4. + 1./2.*jj[j]

print xxCen

for j in range(n):
    for i in range(m):
        xx[0,i,j] = xxCen[0,j] + xxRad * math.cos(xxAng[i])
        xx[1,i,j] = xxCen[1,j] + xxRad * math.sin(xxAng[i])

nw = 2
scale = np.random.normal(size=2*m*n*nw)
xxw = np.empty([2,m,n,nw])

for w in range(nw):
    for j in range(n):
        for i in range(m):
            for l in range(2):
                xxw[l,i,j,w] = xx[l,i,j]+1.E-2*scale[l+i*2+j*2*m+w*2*m*n]


print xx
for w in range(nw):
    for j in range(n):
        plt.plot(xxw[0,:,j,w],xxw[1,:,j,w],'o')

#plt.show()
plt.savefig("initial"+".png")
