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
totSourceCounts = 0
totModel = 0
totNSModel = 0
totPLModel = 0
totBins = 0

totPSRW = 0
totPSRChi = 0
totPSRCounts = 0
totPSRSourceCounts = 0
totPSRModel = 0
totPSRNSModel = 0
totPSRPLModel = 0
totPSRBins = 0

for i in range(nspec/2):
    totPSRBins += len(spcs[i][2])
    totPSRCounts += np.sum(spcs[i][2])
    totPSRSourceCounts += np.sum(spcs[i][2]-spcs[i][3])
    totPSRModel += np.sum(spcs[i][4])
    totPSRNSModel += np.sum(spcs[i][5])
    totPSRPLModel += np.sum(spcs[i][6])
    totPSRW += np.sum(abs(spcs[i][8]))
    totPSRChi += np.sum(abs(spcs[i][9]))

totPWNW = 0
totPWNChi = 0
totPWNCounts = 0
totPWNSourceCounts = 0
totPWNModel = 0
totPWNNSModel = 0
totPWNPLModel = 0
totPWNBins = 0

for i in range(nspec/2):
    totPWNBins += len(spcs[i+nspec/2][2])
    totPWNCounts += np.sum(spcs[i+nspec/2][2])
    totPWNSourceCounts += np.sum(spcs[i+nspec/2][2]-spcs[i+nspec/2][3])
    totPWNModel += np.sum(spcs[i+nspec/2][4])
    totPWNNSModel += np.sum(spcs[i+nspec/2][5])
    totPWNPLModel += np.sum(spcs[i+nspec/2][6])
    totPWNW += np.sum(abs(spcs[i+nspec/2][8]))
    totPWNChi += np.sum(abs(spcs[i+nspec/2][9]))

for i in range(nspec):
    totBins += len(spcs[i][2])
    totCounts += np.sum(spcs[i][2])
    totSourceCounts += np.sum(spcs[i][2]-spcs[i][3])
    totModel += np.sum(spcs[i][4])
    totNSModel += np.sum(spcs[i][5])
    totPLModel += np.sum(spcs[i][6])
    totW += np.sum(abs(spcs[i][8]))
    totChi += np.sum(abs(spcs[i][9]))
    print "spectrum number           -- %i"%(i)
    print "Number of bins            -- %2.0f"%(len(spcs[i][2]))
    print "Number of counts          -- %2.0f"%(np.sum(spcs[i][2]))
    print "Number of source counts   -- %2.0f"%(np.sum(spcs[i][2]-spcs[i][3]))
    print "Number of model counts    -- %2.0f"%(np.sum(spcs[i][4]))
    print "Number of thermal counts  -- %2.0f"%(np.sum(spcs[i][5]))
    print "Number of pl counts       -- %2.0f"%(np.sum(spcs[i][6]))
    print "W Statistic               -- %2.0f"%(np.sum(abs(spcs[i][8])))
    print "Chi-Squared Statistic     -- %2.0f"%(np.sum(abs(spcs[i][9])))

print "Summing over all the spectra:"

print "Total number of bins          -- %2.0f"%(totBins)
print "Total number of counts        -- %2.0f"%(totCounts)
print "Total number of source counts -- %2.0f"%(totSourceCounts)
print "Total number of model counts  -- %2.0f"%(totModel)
print "Number of thermal counts      -- %2.0f"%(totNSModel)
print "Number of pl counts           -- %2.0f"%(totPLModel)
print "Total W Statistic             -- %2.0f"%(totW)
print "Total Chi-Squared Statistic   -- %2.0f"%(totChi)

specs = ['pn','MOS1','MOS2','Total','pn','MOS1','MOS2','Total','Total']

prst = []
lst = []
for i in range(nspec/2):
    lst.append(r'$%i$'%len(spcs[i][2]))
lst.append(r'$%i$'%totPSRBins)
for i in range(nspec/2):
    lst.append(r'$%i$'%len(spcs[i+nspec/2][2]))
lst.append(r'$%i$'%totPWNBins)
lst.append(r'$%i$'%totBins)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][2]))
lst.append(r'$%2.0f$'%totPSRCounts)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i+nspec/2][2]))
lst.append(r'$%2.0f$'%totPWNCounts)
lst.append(r'$%2.0f$'%totCounts)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][2]-spcs[i][3]))
lst.append(r'$%2.0f$'%totPSRSourceCounts)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i+nspec/2][2]-spcs[i+nspec/2][3]))
lst.append(r'$%2.0f$'%totPWNSourceCounts)
lst.append(r'$%2.0f$'%totSourceCounts)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][4]))
lst.append(r'$%2.0f$'%totPSRModel)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i+nspec/2][4]))
lst.append(r'$%2.0f$'%totPWNModel)
lst.append(r'$%2.0f$'%totModel)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][5]))
lst.append(r'$%2.0f$'%totPSRNSModel)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i+nspec/2][5]))
lst.append(r'$%2.0f$'%totPWNNSModel)
lst.append(r'$%2.0f$'%totNSModel)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i][6]))
lst.append(r'$%2.0f$'%totPSRPLModel)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(spcs[i+nspec/2][6]))
lst.append(r'$%2.0f$'%totPWNPLModel)
lst.append(r'$%2.0f$'%totPLModel)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(abs(spcs[i][8])))
lst.append(r'$%2.0f$'%totPSRW)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(abs(spcs[i+nspec/2][8])))
lst.append(r'$%2.0f$'%totPWNW)
lst.append(r'$%2.0f$'%totW)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(abs(spcs[i][9])))
lst.append(r'$%2.0f$'%totPSRChi)
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%np.sum(abs(spcs[i+nspec/2][9])))
lst.append(r'$%2.0f$'%totPWNChi)
lst.append(r'$%2.0f$'%totChi)
prst.append(lst)
lst = []
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%(len(spcs[i][2])-7))
lst.append(r'$%2.0f$'%(totPSRBins-7))
for i in range(nspec/2):
    lst.append(r'$%2.0f$'%(len(spcs[i+nspec/2][2])-7))
lst.append(r'$%2.0f$'%(totPWNBins-7))
lst.append(r'$%2.0f$'%(totBins-7))
prst.append(lst)
lst = []

print prst
print prst[0]

data = {r'ASpectrum': specs, r'Bins': prst[0], r'Counts': prst[1], r'DCounts-Background': prst[2], r'EModel': prst[3], r'FNS Model': prst[4], r'GPL Model': prst[5], r'H$W$': prst[6], r'J$\chi^{2}$': prst[7], r'K$d.o.f$': prst[8] }
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*', 'units':{'$N_{\rm H}$':'$\rm 10^{21} ./ cm^{-2}$'}})
