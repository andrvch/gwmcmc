#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from cudakde import *

infl = sys.argv[1]
otfl = sys.argv[2]
indx = int(sys.argv[3])

samples = read_data(sys.argv[1])

f = open(otfl, "w")

n = shape(samples)[1]
print n
for i in range(n):
    f.write("%.15E "%(samples[indx,i]))
    f.write("\n")
f.close()
