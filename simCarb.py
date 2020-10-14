#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import math
import numpy as np
from xspec import *

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

dist = 1. # kpc

name = sys.argv[1]
time = float(sys.argv[2])

AllData(name+".pi")
AllModels += "carbatm*phabs"
pars = (1.0, 1.4, 13., 100./dist**2, 0.15)
AllModels(1).setPars(pars)

if time == 0:
    fs1 = FakeitSettings(fileName="carbatm_test.fak")
else:
    fs1 = FakeitSettings(fileName="carbatm_test.fak",exposure=time,backExposure=time)

AllData.fakeit(1, fs1)
