#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

#Xset.chatter = 0
Xset.abund = "angr"
Xset.xsect = "bcmc"

name = sys.argv[1]

AllData(name)
AllData(1).ignore("**-0.3")
AllData(1).ignore("10.-**")
AllData(1).ignore("bad")

AllModels += "(bbodyrad+powerlaw)*phabs"
pars = (0.1, 10**(-2.9*2)*10**8, 1.5, 10**-5.1, 0.12)
AllModels(1).setPars(pars)

Fit.statMethod = "cstat"
Fit.statTest = "chi"
Fit.query = "yes"
#Fit.perform()
#Fit.steppar("1 0.05 0.2 10")
#Fit.perform()
#Fit.error("6.63 1-3")

print "--------------------------------------"
print Fit.statistic
print Fit.testStatistic
