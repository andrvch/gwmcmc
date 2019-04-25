#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from xspec import *

erange = [0.3, 7.0]

SPECNAME = "PN_J0633_15asec_grp15.pi"
AllData(SPECNAME)
s1 = AllData(1)

s1.background = " "

AllData.ignore("**-%2.1f %2.1f-**"%(erange[0],erange[1]))
#AllData.ignore("bad")

n = len(s1.values)

for i in range(n):
    print s1.values[i]*3.06E4

print len(s1.values)
