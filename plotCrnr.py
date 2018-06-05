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

fig = corner.corner(samples, no_fill_contours=False, draw_datapoints=True)
fig.savefig("crnr.eps")
