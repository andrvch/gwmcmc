#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

name = sys.argv[1]
time = float(sys.argv[2])
num = int(sys.argv[3])
AllData(name+".pi")
AllModels += "(bbodyrad+powerlaw)*phabs"
pars = (0.1, 10**(-3.*2)*10**8, 1.5, 10**-4., 0.2)
AllModels(1).setPars(pars)
for i in range(num):
    if time == 0:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak")
    else:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak",exposure=time,backExposure=time)
    AllData.fakeit(1, fs1)
