#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

name = sys.argv[1]
bname = sys.argv[2]
time = float(sys.argv[3])
num = int(sys.argv[4])
AllData(name+".pi")
AllModels += "(nsa+powerlaw)*phabs"
pars = (6.1, 1.4, 13., 1.E12, 1.E-6, 1.3, 4.E-5, 0.15)
AllModels(1).setPars(pars)
for i in range(num):
    if time == 0:
        fs1 = FakeitSettings(fileName=name+"_VelaJr"+"%i"%(i)+".fak",background=bname)
    else:
        fs1 = FakeitSettings(fileName=name+"_VelaJr"+"%i"%(i)+".fak",background=bname,exposure=time,backExposure=time)
    AllData.fakeit(1, fs1)
