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

def logistic(x,a,b,g):
    return b/(1+exp(g*(x-a)))

a = 0.1
b = 10.
g = 10.
n = 1000

xx = np.linspace(1.,10.,n)
yy = np.empty([n])
for i in range(n):
    yy[i] = logistic(xx[i],a,b,g)

plt.plot(xx,yy)

#plt.show()
plt.savefig("potential"+".png")
