#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from astropy.io import fits

hdu = fits.open(sys.argv[1])
print hdu[1].header['BACKSCAL']
