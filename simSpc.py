#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

AllModels.lmod("mypackage",dirPath ="/home/aa/nsmdip/")
Xset.addModelString("NSMDIP","/home/aa/nsmdip/")

name = sys.argv[1]
time = float(sys.argv[2])
num = int(sys.argv[3])
AllData(name+".pi")

AllModels += "(nsmdip+powerlaw)*phabs"
pars = (1, 5.3, 0.05, 0., 1., 1.7, 5.E-6, 0.05)
#AllModels += "nsa*phabs"
#pars = (5.7, 1.4, 13., 1.E12, 1.E-6, 0.15)
AllModels(1).setPars(pars)
for i in range(num):
    if time == 0:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak")
    else:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak",exposure=time,backExposure=time)
    AllData.fakeit(1, fs1)
