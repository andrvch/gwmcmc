#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import numpy as np
import torch
import cupy as cp

x = torch.rand(5, 3)
