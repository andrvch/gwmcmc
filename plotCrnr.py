#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cudakde
import corner

samples = np.transpose(cudakde.read_data(sys.argv[1]))
print samples.shape
msamples = samples[:,np.r_[0:7, 13:samples.shape[1]]]
print msamples.shape
fsamples = msamples[np.where(msamples[:,-1]<14000),:]
print fsamples.shape

fig = corner.corner(msamples[np.where(msamples[:,-1]<14000)[0],:], no_fill_contours=False, draw_datapoints=True)
fig.savefig("crnr.eps")
