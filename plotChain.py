#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from cudakde import *

smpls = read_data(sys.argv[1]+"%i"%(int(sys.argv[2]))+".chain")
nwlkrs = int(sys.argv[3])

nprmtrs = shape(smpls)[0]
nstps = int(shape(smpls)[1]/float(nwlkrs))

print(nstps, nprmtrs)

wlkrs = np.empty([nprmtrs,nwlkrs,nstps])

for i in range(nstps):
    for j in range(nwlkrs):
        for k in range(nprmtrs):
            wlkrs[k,j,i] = smpls[k,j+nwlkrs*i]

stps = linspace(1,nstps,num=nstps)
fig, ax = plt.subplots(nrows=nprmtrs)
plt.subplots_adjust(hspace=0.1)

for i in range(nprmtrs):
    for j in range(nwlkrs):
        ax[i].errorbar(stps,wlkrs[i,j,:])

#plt.show()
plt.savefig(sys.argv[1]+"%i"%(int(sys.argv[2]))+"_chain"+".png")
