#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import ticker
from xspec import *
from cudakde import *

def readspectra(FileName):
    lines = []
    with open(FileName) as fp:
        for line in iter(fp.readline, ''):
            lines.append(str(line))
    nsmpl = len(lines)
    spcs = []
    for i in range(nsmpl):
        nbins = len(lines[i].split())
        pars = np.empty([nbins])
        for j in range(nbins):
            pars[j] = lines[i].split()[j]
        spcs.append(pars)
    return spcs


nspec = 6
spcs = readspectra(sys.argv[1])
nnn = int(len(spcs)/6)
print nnn

nbins = len(spcs[0])
xxbins = np.linspace(0,nbins,nbins)

fig, ax = plt.subplots(ncols=1, nrows=2)

ax[0].plot(xxbins,spcs[0])
ax[0].plot(xxbins,spcs[1])
ax[0].plot(xxbins,spcs[2])
ax[1].plot(xxbins,spcs[3])

ax[0].set_yscale('log')

#plt.savefig(sys.argv[2]+".spectra"+".pdf")
plt.show()
