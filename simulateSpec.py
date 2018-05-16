#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

os.system('rm *.fak')

SPECNAME1 = "pwnj0633" #.pi"
AllData(SPECNAME1+".pi")
s1 = AllData(1)
AllModels += "phabs*(powerlaw)" #+bbodyrad)"
m1 = AllModels(1)
pars = (0.1, 1.7, 10**-3.) #, 0.13, 10**-3.4*10**8)
m1.setPars(pars)
fs1 = FakeitSettings(fileName=SPECNAME1+"1"+".fak",exposure=1.E6)
#fs1 = FakeitSettings()
AllData.fakeit(1, fs1)

SPECNAME2 = "pwnj0633"
AllData(SPECNAME2+".pi")
s1 = AllData(1)
AllModels += "phabs*(powerlaw)"
m1 = AllModels(1)
pars = (0.1, 1.7, 10**-3.)
m1.setPars(pars)
fs1 = FakeitSettings(fileName=SPECNAME2+"2"+".fak",exposure=1.E6)
#fs1 = FakeitSettings()
AllData.fakeit(1, fs1)
