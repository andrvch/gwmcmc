#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import torch

x = torch.rand(5, 3)
print(x)
