
import os, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *

f = open(sys.argv[1])
lines = f.readlines()
ChainOut = []
for i in range(len(lines)):
    ChainOut.append(np.array([float(x) for x in lines[i].split()]))

nwalkers = int(len(ChainOut[0])/6)
nsteps   = len(ChainOut)

Nh = np.empty([nwalkers,nsteps])
Ga = np.empty([nwalkers,nsteps])
Np = np.empty([nwalkers,nsteps])
Te = np.empty([nwalkers,nsteps])
Nn = np.empty([nwalkers,nsteps])
St = np.empty([nwalkers,nsteps])

for i in range(nwalkers):
    for j in range(nsteps):
        Nh[i,j] = ChainOut[j][6*i]
        Ga[i,j] = ChainOut[j][6*i+1]
        Np[i,j] = ChainOut[j][6*i+2]
        Te[i,j] = ChainOut[j][6*i+3]
        Nn[i,j] = ChainOut[j][6*i+4]
        St[i,j] = ChainOut[j][6*i+5]

steps = np.linspace(0,nsteps-1,num=nsteps)

fig, ax = plt.subplots(nrows=6)
plt.subplots_adjust(hspace=0.1)

for j in range(nwalkers):
    ax[0].errorbar(steps, Nh[j,:])
    ax[1].errorbar(steps, Ga[j,:])
    ax[2].errorbar(steps, Np[j,:])
    ax[3].errorbar(steps, Te[j,:])
    ax[4].errorbar(steps, Nn[j,:])
    ax[5].errorbar(steps, St[j,:])

setp([a.get_xticklabels() for a in ax[:5]], visible=False)
plt.show()
