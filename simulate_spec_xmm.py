#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from xspec import *

os.system('rm *.fak')

SPECNAME = "PN_J0633_15asec_grp0.pi"
AllData(SPECNAME)
s1 = AllData(1)

AllModels += "phabs*powerlaw"
m1 = AllModels(1)
pars = (0.15, 1.5, 1.)

m1.setPars(pars)

fs1 = FakeitSettings()
AllData.fakeit(1, fs1)
