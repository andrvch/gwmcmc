#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pylab import *
import numpy as np

pipi = 3.14159265359

N = 1000

E0 = 1.
W = 100.
l2 = 1.
l3 = 1.
bt = 1.
bt3 = 1.

#lmbd = math.sqrt(l2*l1*(1+bt**2)/(1+bt3**2))
#lmbd3 = 64.*l3/55./pipi/(1+bt3**2)

yy = np.linspace(-W/2.,W/2.,N)

def velocity(y,E0,W,l2,l3,bt,bt3):
    lmbd = math.sqrt(l2*l3*(1+bt**2)/(1+bt3**2))
    lmbd3 = 64.*l3/55./pipi/(1+bt3**2)
    print(lmbd)
    print(lmbd3)
    A = bt*E0*W/2./(1+lmbd3/lmbd)*math.exp(-W/2./lmbd)
    print(A)
    D = E0*W**2/8./l2 + 3*pipi*E0*W/16. + (bt3-bt)*E0*l3*bt*W/2./(1+bt3**2)/(lmbd+lmbd3)
    print(D)
    v = - E0*y**2/2./l2 + D + l3*(bt+bt3)/lmbd/(1+bt3**2)*2*A*math.cosh(y/lmbd)
    print(v)
    return v

def velocity_poi(y,E0,W,l2,l3,bt,bt3):
    lmbd = math.sqrt(l2*l3*(1+bt**2)/(1+bt3**2))
    lmbd3 = 64.*l3/55./pipi/(1+bt3**2)
    print(lmbd)
    print(lmbd3)
    A = bt*E0*W/2./(1+lmbd3/lmbd)*math.exp(-W/2./lmbd)
    print(A)
    D = E0*W**2/8./l2 #+ 3*pipi*E0*W/16. + (bt3-bt)*E0*l3*bt*W/2./(1+bt3**2)/(lmbd+lmbd3)
    print(D)
    v = - E0*y**2/2./l2 + D #+ l3*(bt+bt3)/lmbd/(1+bt3**2)*2*A*math.cosh(y/lmbd)
    print(v)
    return v

vv = np.empty([N])
vvp = np.empty([N])

for i in range(N):
    vv[i] = velocity(yy[i],E0,W,l2,l3,bt,bt3)
    vvp[i] = velocity_poi(yy[i],E0,W,l2,l3,bt,bt3)
    #print(vv[i])

plt.plot(yy,vv,'-')
plt.plot(yy,vvp,'--')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.savefig("fig3"+".png")
