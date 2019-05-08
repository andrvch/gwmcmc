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
AllModels += "(nsa+powerlaw)*gabs*phabs"
pars = (6.0, 1.4, 13., 1.E12, 1.E-6, 1.5, 4.E-5, 0.8, 0.05, 0.01, 0.15)
AllModels(1).setPars(pars)
for i in range(num):
    if time == 0:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak")
    else:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak",exposure=time,backExposure=time)
    AllData.fakeit(1, fs1)
