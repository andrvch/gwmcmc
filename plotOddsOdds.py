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

mm = [2,3,4,5,6]

fig, ax = plt.subplots()

plt.setp(ax.get_yticklabels(), fontsize=12)
plt.setp(ax.get_xticklabels(), fontsize=12)

ax.plot(mm,samples[0],'-o')

plt.xticks(mm)
plt.xlabel(r'$m$',fontsize=14)
plt.ylabel(r'$O_{m1}$',fontsize=14)
#plt.show()
plt.savefig(sys.argv[1]+"oddsODS"+".eps")
