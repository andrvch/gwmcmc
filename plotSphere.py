#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sphvol(x,N):
    return x**N

x = np.linspace(0.,1.,1000)

for i in (3,10,100,1000):
    plt.plot(x,sphvol(x,i),label="N=%i"%(i))
#plt.set_xticklabels
plt.tick_params(labelsize=14)
plt.legend(fontsize=16)
plt.xlabel(r"$r$",fontsize=16)
plt.ylabel(r"$r^{N}$",fontsize=16)
plt.savefig("sphere.pdf")

#plt.show()
