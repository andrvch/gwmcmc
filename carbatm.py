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

ACONST = 2.

ndm = 100
nwl = 512 #sys.argv[1]
nst = 2048
x0 = 1.
dlt = 0.02

# kernel to compute statistic
l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

# set a seed for random numbers
cp.random.seed(123)

#chn = cp.empty((nst,nwl,ndm),dtype=np.float32)
#stt = cp.empty((nst,nwl),dtype=np.float32)

# initialize walkers
xx = cp.full((nwl,ndm),x0,dtype=np.float32) + dlt*cp.random.randn(nwl,ndm,dtype=np.float32)
#print(xx)
#print(xx.shape[0])

# initialize statistic
st = cp.empty(nwl,dtype=np.float32)
#st = l2norm_kernel(xx, axis=1)
st = cp.sum(cp.power(xx,2.),axis=1)

# initialize arrays to store results
chn = cp.stack([xx])
stt = cp.stack([st])
#print(chn)
#print(stt)
print(chn.shape)
print(stt.shape)

# generate random walker indices at each step
arrays = [cp.random.random_integers(0, int(nwl/2)-1, size=nwl) for i in range(nst)]
wind = cp.stack(arrays,axis=0)
#print(wind)
#print(wind.shape[0])

# generate z random numbers
# distributed as 1/sqrt(z) in the range [1/a,a], a=ACONST
zz = cp.random.rand(nst,nwl,dtype=np.float32)
zz = 1./ACONST*cp.power(zz*(ACONST-1.)+1,2.)
#print(zz)

# generate r random numbers
rr = cp.random.rand(nst,nwl,dtype=np.float32)
#print(rr)

for i in range(nst):
    xx0 = xx[:int(nwl/2),:]
    st0 = st[:int(nwl/2)]
    xxC = xx[int(nwl/2):,:]
    stC = st[int(nwl/2):]
    inC = wind[i,int(nwl/2):]
    zz0 = zz[i,:int(nwl/2)]
    rr0 = rr[i,:int(nwl/2)]
    ##
    xxCP = xxC[inC,:]
    xx1 = xxCP + (xx0 - xxCP)*zz0[:,None]
    #st1 = l2norm_kernel(xx1, axis=1)
    # compute statistic:
    st1 = cp.sum(cp.power(xx1,2.),axis=1)
    qq = cp.exp(-0.5 * (st1 - st0)) * cp.power(zz0,ndm - 1)
    more = cp.array(qq>rr0,dtype=np.int32)
    less = cp.array(qq<=rr0,dtype=np.int32)
    #more = cp.greater(qq,rr0,dtype=np.int32)
    #less = cp.less_equal(qq,rr0,dtype=np.int32)
    xx0 = more[:,None]*xx1 + less[:,None]*xx0
    st0 = more*st1 + less*st0
    ##
    xx = cp.append(xx0,xxC,axis=0)
    st = cp.append(st0,stC)
    ##
    xx0 = xx[int(nwl/2):,:]
    st0 = st[int(nwl/2):]
    xxC = xx[:int(nwl/2),:]
    stC = st[:int(nwl/2)]
    inC = wind[i,:int(nwl/2)]
    zz0 = zz[i,int(nwl/2):]
    rr0 = rr[i,int(nwl/2):]
    ##
    xxCP = xxC[inC,:]
    xx1 = xxCP + (xx0 - xxCP)*zz0[:,None]
    #st1 = l2norm_kernel(xx1, axis=1)
    # compute statistic
    st1 = cp.sum(cp.power(xx1,2.),axis=1)
    qq = cp.exp(-0.5 * (st1 - st0)) * cp.power(zz0,ndm - 1)
    more = cp.array(qq>rr0,dtype=np.int32)
    less = cp.array(qq<=rr0,dtype=np.int32)
    #more = cp.greater(qq,rr0,dtype=np.int32)
    #less = cp.less_equal(qq,rr0,dtype=np.int32)
    xx0 = more[:,None]*xx1 + less[:,None]*xx0
    st0 = more*st1 + less*st0
    ##
    xx = cp.append(xxC,xx0,axis=0)
    st = cp.append(stC,st0,axis=0)
    ##
    chn = cp.append(chn,cp.stack([xx]),axis=0)
    stt = cp.append(stt,cp.stack([st]),axis=0)

# copy results to cpu
chn_cpu = cp.asnumpy(chn)
stt_cpu = cp.asnumpy(stt)

# plot chain
stps = linspace(1,nst,num=nst+1)
fig, ax = plt.subplots(nrows=4)
plt.subplots_adjust(hspace=0.1)

for i in range(3):
    for j in range(nwl):
        ax[i].errorbar(stps,chn_cpu[:,j,i])
for j in range(nwl):
    ax[3].errorbar(stps,stt_cpu[:,j])

#plt.show()
plt.savefig("simplegauss_chain"+".png")
