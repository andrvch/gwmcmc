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

nbins1D = 100
nbins2D = 200

samples = read_data(sys.argv[1])
#print samples.shape
#samples = samples[:samples.shape[0],:]
print samples.shape
#samples = samples[:,np.where(samples[-1,:]<14000)[0]]
#print samples.shape

#odds = - 0.5 * samples[-1] + 1717 + 1717 * math.log(9.1917e4) + ( 5. / 2. ) * math.log(9.1917e4) + 1717 * math.log(5) + 4. * math.log(2.*3.14) - 0.5*math.log(1717) - 2.84128601128261974624e1
odds = -0.5 * samples[-1]
frqs = samples[0]
plt.plot(frqs,odds,'o')

oddsN = np.exp(odds)/frqs
print oddsN.sum()/len(oddsN)
#plt.show()
plt.savefig(sys.argv[1]+"odds"+".jpg")
