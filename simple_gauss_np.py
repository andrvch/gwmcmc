#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import os, sys
import math
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

ACONST = 2.

ndm = 2
nwl = 512 #sys.argv[1]
nst = 8192
x0 = 1.
dlt = 0.02

cp.random.seed(123)

#chn = cp.empty((nst,nwl,ndm),dtype=np.float32)
#stt = cp.empty((nst,nwl),dtype=np.float32)

# initialize walkers
xx = cp.full((nwl,ndm),x0,dtype=np.float32) + dlt*cp.random.randn(nwl,ndm,dtype=np.float32)
#print(xx)
#print(xx.shape[0])
#sstt = cp.empty(ndm*nwl,dtype=np.float32)
st = cp.empty(nwl,dtype=np.float32)

l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

# initialize statistic
st = l2norm_kernel(xx, axis=1)

chn = cp.stack([xx])
stt = cp.stack([st])
print(chn)
print(stt)
print(chn.shape)
print(stt.shape)

# generate random walker indices at each step
arrays = [cp.random.random_integers(0, int(nwl/2)-1, size=nwl) for i in range(nst)]
wind = cp.stack(arrays,axis=0)
#print(wind)
#print(wind.shape[0])

# generate z random numbers
# distributed as 1/sqrt(z) in the range [1/a,a]
zz = cp.random.rand(nst,nwl,dtype=np.float32)
zz = 1./ACONST*cp.power(zz*(ACONST-1.)+1,2.)
#print(zz)

# generate r random numbers
rr = cp.random.rand(nst,nwl,dtype=np.float32)
#print(rr)

xx0 = xx[:int(nwl/2),:]
st0 = st[:int(nwl/2)]
#print(xx0)
xxC = xx[int(nwl/2):,:]
#print(xxC)
xxCP = xxC[wind[0,int(nwl/2):],:]
#print(xxCP)
xx1 = xxCP + (xx0 - xxCP)*zz[0,:int(nwl/2),None]
#print(xx1)
st1 = l2norm_kernel(xx1, axis=1)
qq = cp.exp(-0.5 * (st1 - st0)) * cp.power(zz[0,:int(nwl/2)],ndm - 1)
#print(qq)
#print(rr[0,:int(nwl/2)])
comp = cp.empty(int(nwl/2),dtype=np.int32)
more = cp.array(qq>rr[0,:int(nwl/2)],dtype=np.int32)
less = cp.array(qq<=rr[0,:int(nwl/2)],dtype=np.int32)
#print(more)
#print(less)
xx0 = more[:,None]*xx1 + less[:,None]*xx0
#print(xx0)
#print(xxC)
xx = cp.append(xx0,xxC,axis=0)
#print(xx)

#chn = cp.append(chn,cp.stack([xx]),axis=0)
#print(cp.stack([xx]))
#print(xx)
#print(chn)
#print(chn.shape)
#xxD = xx0 - xxC
#print(xxD)
#print(zz[0,:int(nwl/2)])
#xxW = xxD * zz[0,:int(nwl/2),None]
#print(xxW)

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
    st1 = l2norm_kernel(xx1, axis=1)
    qq = cp.exp(-0.5 * (st1 - st0)) * cp.power(zz0,ndm - 1)
    more = cp.array(qq>rr0,dtype=np.int32)
    less = cp.array(qq<=rr0,dtype=np.int32)
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
    st1 = l2norm_kernel(xx1, axis=1)
    qq = cp.exp(-0.5 * (st1 - st0)) * cp.power(zz0,ndm - 1)
    more = cp.array(qq>rr0,dtype=np.int32)
    less = cp.array(qq<=rr0,dtype=np.int32)
    xx0 = more[:,None]*xx1 + less[:,None]*xx0
    st0 = more*st1 + less*st0
    ##
    xx = cp.append(xxC,xx0,axis=0)
    st = cp.append(stC,st0,axis=0)
    ##
    chn = cp.append(chn,cp.stack([xx]),axis=0)
    stt = cp.append(stt,cp.stack([st]),axis=0)

chn_cpu = cp.asnumpy(chn)
stt_cpu = cp.asnumpy(stt)

stps = linspace(1,nst,num=nst+1)
fig, ax = plt.subplots(nrows=ndm+1)
plt.subplots_adjust(hspace=0.1)

for i in range(ndm):
    for j in range(nwl):
        ax[i].errorbar(stps,chn_cpu[:,j,i])
for j in range(nwl):
    ax[ndm].errorbar(stps,stt_cpu[:,j])
#plt.show()
plt.savefig("simplegauss_chain"+".png")

#cp.random.default_rng.shuffle(wind,axis=1)
#rn = cp.random.rand(int(3*nst*nwl),dtype=np.float32)
#print(rn)
