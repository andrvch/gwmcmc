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

samples1 = read_data(sys.argv[1])
samples2 = read_data(sys.argv[2])
print samples1.shape

m = int(sys.argv[3])

ph = np.linspace(0.,1.,num=m+1)
ph1 =  np.linspace(1.,2.,num=m+1)

phase = np.concatenate((ph[:-1],ph1[:-1]))

fig, ax = plt.subplots(nrows=2)

pp = np.empty([m,2])
pp1 = np.empty([m,2])
nn0 = np.empty([m,2])
for i in range(2):
    for j in range(m):
        pp[j,i] = ph[j+i]
        pp1[j,i] = ph1[j+i]
        nn0[j,i] = samples1[1,2+j]

ss = np.concatenate((samples1[1,2:2+m],samples1[1,2:2+m]))
ss1 = np.concatenate((samples1[1,2:2+m]-samples1[0,2:2+m],samples1[1,2:2+m]-samples1[0,2:2+m]))
ss2 = np.concatenate((samples1[2,2:2+m]-samples1[1,2:2+m],samples1[2,2:2+m]-samples1[1,2:2+m]))

ax[0].plot(pp[0],nn0[0],color='black')

ax[0].step(phase+1./float(m),ss,color='black')
ax[0].errorbar(phase+1./float(m)/2.,ss,yerr=[ss1,ss2],fmt=' ',color='black')
#ax[0].set_xlabel("phase bins",fontsize=14)
ax[0].set_ylabel("Photons",fontsize=14)

plt.setp(ax[0].get_yticklabels(), fontsize=14)
plt.setp(ax[0].get_xticklabels(), fontsize=14)
plt.setp(ax[0].get_xticklabels(), visible=False)

for i in range(2):
    for j in range(m):
        nn0[j,i] = samples2[1,2+j]

ss = np.concatenate((samples2[1,2:2+m],samples2[1,2:2+m]))
ss1 = np.concatenate((samples2[1,2:2+m]-samples2[0,2:2+m],samples2[1,2:2+m]-samples2[0,2:2+m]))
ss2 = np.concatenate((samples2[2,2:2+m]-samples2[1,2:2+m],samples2[2,2:2+m]-samples2[1,2:2+m]))

ax[1].plot(pp[0],nn0[0],color='black')

ax[1].step(phase+1./float(m),ss,color='black')
ax[1].errorbar(phase+1./float(m)/2.,ss,yerr=[ss1,ss2],fmt=' ',color='black')
ax[1].set_xlabel("Phase",fontsize=14)
ax[1].set_ylabel("Photons",fontsize=14)

plt.setp(ax[1].get_yticklabels(), fontsize=14)
plt.setp(ax[1].get_xticklabels(), fontsize=14)

plt.savefig(sys.argv[1]+"curve"+".eps")
