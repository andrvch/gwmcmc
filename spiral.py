#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

def xx(y,R):
    logy = np.log(y)
    zz = R**2-y**2+2*logy+1.
    return np.sqrt(zz)

def xixixi(xi,R):
    zz = R + math.sqrt(R**2-1)*np.cos(xi)
    return zz

R = 5.
RR = np.linspace(1.5,10.,10)

for i in range(len(RR)):
    xixi = np.linspace(0.,3./2.,1000)
    yy = xixixi(xixi,RR[i])
    plt.plot(yy,xx(yy,RR[i]))

plt.savefig("spiral.jpg")
#plt.show()
