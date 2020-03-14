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

AllModels.lmod("mypackage",dirPath ="/home/aa/nsmdip/")
Xset.addModelString("NSMDIP","/home/aa/nsmdip/")

AllModels += "nsmdip*phabs"
pars = (1, 5.2, 1., 0., 1., 0.05)
AllModels(1).setPars(pars)
for i in range(num):
    fs1 = FakeitSettings(response=name+".rmf", arf=name+".arf", fileName=name+"_"+"%1i"%(i)+".fak", exposure=time)
    AllData.fakeit(1, fs1)
