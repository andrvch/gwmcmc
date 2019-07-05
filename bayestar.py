#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import dustmaps
import dustmaps.bayestar
from dustmaps.config import config
config['data_dir'] = '/data/aa/abcdustmaps'

dustmaps.bayestar.fetch()
