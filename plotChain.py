#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from astropy.io import ascii

chain = ascii.read(sys.argv[1],format='no_header')
print chain.indices
print chain[0][0]
#print chain[0][2]
