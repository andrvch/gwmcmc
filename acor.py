#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import os, sys
import math
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from astropy.io import ascii
from astropy.io import fits
