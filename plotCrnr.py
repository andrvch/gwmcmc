#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cudakde
import corner

#nsm = 500000
#samples = cudakde.read_data_nsmpl(sys.argv[1],nsm)
samples = cudakde.read_data(sys.argv[1])
print samples.shape
samples = samples[np.r_[0:1, 2:5, 11:samples.shape[0]],:]
print samples.shape
#samples = samples[:,np.where(samples[-1,:]<14000)[0]]
#print samples.shape
samples = np.transpose(samples)
print samples.shape

fig = corner.corner(samples, no_fill_contours=False, draw_datapoints=True)
fig.savefig('psrcrnr.eps')
