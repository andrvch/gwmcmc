#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import cudakde
import corner

#nsm = 500000
#samples = cudakde.read_data_nsmpl(sys.argv[1],nsm)
samples = cudakde.read_data(sys.argv[1])
print samples.shape
#samples = samples[np.r_[0:7, 13:samples.shape[0]-1],:]
#samples = samples[:-1,np.r_[np.where(samples[:,-2]<25240.)]]
samples = samples[:-1,:]
print samples.shape

samples = np.transpose(samples)
print samples.shape

fig = corner.corner(samples, no_fill_contours=False, draw_datapoints=True)
#fig.savefig(sys.argv[1]+"crnrs"+".pdf")
plt.show()
