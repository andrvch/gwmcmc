#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
from pylab import *
import asciitable
import asciitable.latex
from cudakde import *

nspec = 6
spcs = readspectra(nspec,sys.argv[1]+"spec"+".spec")
erange = [0.4,7.0]

totW = 0
totChi = 0
totCounts = 0
totModel = 0
totSourceCounts = 0
totBins = 0

for i in range(nspec):
    totBins += len(spcs[i][2])
    totCounts += np.sum(spcs[i][2])
    totModel += np.sum(spcs[i][4])
    totSourceCounts += np.sum(spcs[i][2]-spcs[i][3])
    totW += np.sum(abs(spcs[i][6]))
    totChi += np.sum(abs(spcs[i][7]))
    print "spectrum number           -- %i"%(i)
    print "Number of bins            -- %2.0f"%(len(spcs[i][2]))
    print "Number of counts          -- %2.0f"%(np.sum(spcs[i][2]))
    print "Number of source counts   -- %2.0f"%(np.sum(spcs[i][2]-spcs[i][3]))
    print "Number of model counts    -- %2.0f"%(np.sum(spcs[i][4]))
    print "W Statistic               -- %2.0f"%(np.sum(abs(spcs[i][6])))
    print "Chi-Squared Statistic     -- %2.0f"%(np.sum(abs(spcs[i][7])))

print "Summing over all the spectra:"

print "Total number of bins          -- %2.0f"%(totBins)
print "Total number of counts        -- %2.0f"%(totCounts)
print "Total number of source counts -- %2.0f"%(totSourceCounts)
print "Total number of model counts  -- %2.0f"%(totModel)
print "Total W Statistic             -- %2.0f"%(totW)
print "Total Chi-Squared Statistic   -- %2.0f"%(totChi)

specs = ['pn','MOS1','MOS2','pn','MOS1','MOS2','Total']

prst = []
lst = []
for i in range(nspec):
    lst.append(r'$%i$'%len(spcs[i][2]))
lst.append(r'$%i$'%totBins)
prst.append(lst)
lst = []
for i in range(nspec):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][2]))
lst.append(r'$%2.0f$'%totCounts)
prst.append(lst)
lst = []
for i in range(nspec):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][2]-spcs[i][3]))
lst.append(r'$%2.0f$'%totSourceCounts)
prst.append(lst)
lst = []
for i in range(nspec):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][4]))
lst.append(r'$%2.0f$'%totModel)
prst.append(lst)
lst = []
for i in range(nspec):
    lst.append(r'$%2.0f$'%np.sum(abs(spcs[i][6])))
lst.append(r'$%2.0f$'%totW)
prst.append(lst)
lst = []
for i in range(nspec):
    lst.append(r'$%2.0f$'%np.sum(abs(spcs[i][7])))
lst.append(r'$%2.0f$'%totChi)
prst.append(lst)
lst = []
for i in range(nspec):
    lst.append(r'$%2.0f$'%(len(spcs[i][2])-7))
lst.append(r'$%2.0f$'%(totBins-7))
prst.append(lst)
lst = []

print prst
print prst[0]

data = {r'ASpectrum': specs, r'Bins': prst[0], r'Counts': prst[1], r'DCounts-Background': prst[2], r'EModel': prst[3], r'F$W$': prst[4], r'G$\chi^{2}$': prst[5], r'H$d.o.f$': prst[6] }
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*', 'units':{'$N_{\rm H}$':'$\rm 10^{21} ./ cm^{-2}$'}})
