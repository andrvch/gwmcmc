#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

name = sys.argv[1]
time = float(sys.argv[2])
num = int(sys.argv[3])
AllModels += "phabs*(powerlaw)" #+bbodyrad)"
pars = (0.2, 1.7, 10**-3.) #, 0.13, 10**-3.4*10**8)
AllModels(1).setPars(pars)
for i in range(num):
    fs1 = FakeitSettings(response=name+".rmf", arf=name+".arf", fileName=name+"_"+"%1i"%(i)+".fak", exposure=time)
    AllData.fakeit(1, fs1)
