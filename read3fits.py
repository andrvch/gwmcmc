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

fileName = sys.argv[1] # (files position_input or position_input_11)

hdul = fits.open(fitsName)
hdul.info()
