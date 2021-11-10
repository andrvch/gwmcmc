#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pylab import *
from astropy.io import ascii
from astropy.io import fits
import numpy as np
#import pyregion
#import re

fl = "spC.fits"

hdul = fits.open(fl)
print(shape(hdul)[0])

for i in range(shape(hdul)[0]):
    print("%i:"%i)
    hdr = hdul[i].header
    #print(repr(hdr))
    data = hdul[i].data
    print(shape(data))
    if i < 4:
        print(data)
