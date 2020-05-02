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
#ii = np.array([0, 0, 1, 1])
#jj = np.array([0, 1, 0, 1])
#ii = np.random.uniform(size=n)
#jj = np.random.uniform(size=n)
#for j in range(n):
#    xxCen[0,j] = ii[j] #1./(n-1.) + 1./(n-1.)*ii[j]
#    xxCen[1,j] = jj[j] #1./(n-1.) + 1./(n-1.)*jj[j]

tt = 0
gg = 0
ii = 0
while  ii < n :
    xxCen[0,ii] = 1. / ( n - 1. ) * ( 1. + tt )
    xxCen[1,ii] = 1. / ( n - 1. ) * ( 1. + gg )
    if gg < n/2.-1.:
        gg += 1
    else:
        tt += 1
        gg = 0
    ii += 1

print xxCen

for j in range(n):
    for i in range(m):
        xx[0,i,j] = xxCen[0,j] + xxRad * math.cos(xxAng[i])
        if xx[0,i,j] > 1.:
            xx[0,i,j] = xx[0,i,j] - 1.
        elif xx[0,i,j] < 0.:
            xx[0,i,j] = xx[0,i,j] + 1.
        xx[1,i,j] = xxCen[1,j] + xxRad * math.sin(xxAng[i])
        if xx[1,i,j] > 1.:
            xx[1,i,j] = xx[1,i,j] - 1.
        elif xx[1,i,j] < 0.:
            xx[1,i,j] = xx[1,i,j] + 1.

nw = 100
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

plt.xlim(0.,1.)
plt.ylim(0.,1.)
#plt.show()
plt.savefig("initial"+".png")
