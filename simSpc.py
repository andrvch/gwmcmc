#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

name = "PN_J0633_15asec_grp15"
time = float(sys.argv[1])
num = int(sys.argv[2])
AllData(name+".pi")
AllModels += "nsa*phabs"
pars = (6.0, 1.4, 13., 1.E12, 1.E-4, 0.15)
AllModels(1).setPars(pars)
for i in range(num):
    if time == 0:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak")
    else:
        fs1 = FakeitSettings(fileName=name+"_"+"%i"%(i)+".fak",exposure=time,backExposure=time)
    AllData.fakeit(1, fs1)
