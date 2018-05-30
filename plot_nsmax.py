#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *

inf = open(sys.argv[1])
lin = inf.readlines()
en = np.array([float(x) for x in lin[1].split()])
nt = len(lin[0].split())
fx = np.empty([nt,len(en)])
for i in range(nt):
  fx[i,:] = np.array([float(x) for x in lin[2+i].split()])
  
inf = open(sys.argv[2])
lin = inf.readlines()
npars = len(lin[3].split()) 
nsmpl = len(lin)      
pars2 = np.empty([npars,nsmpl-3])
for j in range(npars):
  for i in range(3,nsmpl):
    pars2[j,i-3] = lin[i].split()[j]

for i in range(nt):
  plt.plot(en,10**26.178744*fx[i,:]/en,color='b')

for i in range(1,len(pars2)):   
  plt.plot(pars2[0,:],pars2[i,:],color='r')

yscale('log',nonposy='clip')
xscale('log')
plt.show()