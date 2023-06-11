#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import os, sys
import math
import numpy as np
#import torch
import cupy as cp

ndm = 2
nwl = 8 #sys.argv[1]
nst = 128
x0 = -1.
dlt = 0.02

xx = cp.full((ndm,nwl),x0,dtype=np.float32) + dlt*cp.random.randn(ndm,nwl,dtype=np.float32)
print(xx)
sstt = cp.empty((ndm,nwl),dtype=np.float32)

l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

add_kernel = cp.RawKernel(r'''
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
extern "C" __global__ void returnXXStat ( const int dim, const int nwl, const float *xx, float *s ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int t = i + j * dim;
  int t1 = i + j * ( dim - 1 );
  float d = dim * 1.;
  if ( i < dim - 1 && j < nwl ) {
    s[t1] = d * pow ( xx[t+1] - xx[t], 2. ) + ( pow ( 1 - pow ( xx[t+1], 2. ), 2. ) + pow ( 1 - pow ( xx[t], 2. ), 2. ) ) / d;
  }
}
''', 'returnXXStat')

rn = cp.random.rand(int(2*3*nst*nwl/2.),dtype=np.float32)
#print(rn)
