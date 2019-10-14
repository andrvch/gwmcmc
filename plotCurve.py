#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import *
import random
import time
from cudakde import *

samples = read_data(sys.argv[1])
print samples.shape

m = int(sys.argv[2])

ph = np.linspace(0.,1.,num=m+1)
ph1 =  np.linspace(1.,2.,num=m+1)

nn1 = np.empty([m,2])
nn2 = np.empty([m,2])
pp = np.empty([m,2])
pp1 = np.empty([m,2])
nn0 = np.empty([m,2])
for i in range(2):
    for j in range(m):
        pp[j,i] = ph[j+i]
        pp1[j,i] = ph1[j+i]
        nn1[j,i] = samples[0,2+j]
        nn2[j,i] = samples[2,2+j]
        nn0[j,i] = samples[1,2+j]

ppp = np.concatenate((pp,pp[:,::-1]),axis=1)
ppp1 = np.concatenate((pp1,pp1[:,::-1]),axis=1)
nnn = np.concatenate((nn1,nn2),axis=1)

fig, ax = plt.subplots()

#ax.plot(pp[0],nn0[0],color='black')
phase = np.concatenate((ph[:-1],ph1[:-1]))

ss = np.concatenate((samples[1,2:2+m],samples[1,2:2+m]))
ss1 = np.concatenate((samples[1,2:2+m]-samples[0,2:2+m],samples[1,2:2+m]-samples[0,2:2+m]))
ss2 = np.concatenate((samples[2,2:2+m]-samples[1,2:2+m],samples[2,2:2+m]-samples[1,2:2+m]))

ax.step(phase+1./float(m),ss,color='black')
ax.errorbar(phase+1./float(m)/2.,ss,yerr=[ss1,ss2],fmt=' ',color='black')
ax.set_xlabel("phase bins",fontsize=14)
ax.set_ylabel("number of photons",fontsize=14)

plt.setp(ax.get_yticklabels(), fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=14)

plt.savefig(sys.argv[1]+"curve"+".jpg")
