#!/usr/bin/env python3
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
import pyregion
import re

fl = "spC.fits"

hdul = fits.open(fl)

print("0:")
hdr = hdul[0].header
print(repr(hdr))

print("1:")
hdr = hdul[1].header
print(repr(hdr))

print("2:")
hdr = hdul[2].header
print(repr(hdr))

print("3:")
hdr = hdul[3].header
print(repr(hdr))

print("4:")
hdr = hdul[4].header
print(repr(hdr))

print("5:")
hdr = hdul[5].header
print(repr(hdr))
