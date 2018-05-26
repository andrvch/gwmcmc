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
AllData.ignore("**-0.3")
AllData.ignore("10.-**")
AllData.ignore("bad")

AllModels += "phabs*bbodyrad"
pars = (0.121, 0.131, 10**(-3.15*2)*10**8)
AllModels(1).setPars(pars)

Fit.statMethod = "cstat"
Fit.statTest = "ad"
Fit.query = "yes"
Fit.perform()
