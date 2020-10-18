#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

dist = 1. # kpc

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

name = sys.argv[1]
time = float(sys.argv[2])
num = int(sys.argv[3])

#AllModels += "carbatm*phabs"
#pars = (1.0, 1.4, 13., 100./dist**2, 0.15)
AllModels += "nsa*phabs"
pars = (6.0, 1.4, 13., 1.E12, 1./(dist*1.E3)**2, 0.15)
AllModels(1).setPars(pars)

AllModels(1).setPars(pars)
for i in range(num):
    fs1 = FakeitSettings(response=name+".rmf", arf=name+".arf", fileName="carbatm_test.fak", exposure=time)
    AllData.fakeit(1, fs1)
