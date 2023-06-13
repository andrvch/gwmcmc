#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import os, sys
import math
import numpy as np
#import torch
import cupy as cp

THRDSPERBLCK = 32

def grid2D(n,m):
    return ( int( ( n + THRDSPERBLCK - 1 ) / THRDSPERBLCK ), int( ( m + THRDSPERBLCK - 1 ) / THRDSPERBLCK )  )

print(grid2D(100,128))

def block2D():
    return ( THRDSPERBLCK, THRDSPERBLCK )

print(block2D())

ndm = 10
nwl = 32 #sys.argv[1]
nst = 128
x0 = -1.
dlt = 0.02

xx = cp.full(ndm*nwl,x0,dtype=np.float32) + dlt*cp.random.randn(ndm*nwl,dtype=np.float32)
print(xx)
sstt = cp.empty(ndm*nwl,dtype=np.float32)
stt = cp.empty(nwl,dtype=np.float32)

l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

xxStat_kernel = cp.RawKernel(r'''
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
extern "C" __global__ void xxStat ( const int dim, const int nwl, const float *xx, float *s ) {
  int di = threadIdx.x + blockDim.x * blockIdx.x;
  int dj = threadIdx.y + blockDim.y * blockIdx.y;
  int t = di + dj * dim;
  int t1 = di + dj * ( dim - 1 );
  float d = dim * 1.;
  if ( di < dim - 1 && dj < nwl ) {
    s[t1] = d * pow ( xx[t+1] - xx[t], 2. ) + ( pow ( 1 - pow ( xx[t+1], 2. ), 2. ) + pow ( 1 - pow ( xx[t], 2. ), 2. ) ) / d;
  }
}
''', 'xxStat')

xxStat_kernel(grid2D(ndm-1,nwl),block2D(),(ndm,nwl,xx,sstt))
print(sstt)

rn = cp.random.rand(int(2*3*nst*nwl/2.),dtype=np.float32)
#print(rn)
