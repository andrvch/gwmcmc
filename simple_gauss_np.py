#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pycuda.autoinit
#import pycuda.driver as drv
import os, sys
import math
import numpy as np
#import cupy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

ACONST = 2.

ndm = 100
nwl = 512 #sys.argv[1]
nst = 4096
x0 = 1.
dlt = 0.02

np.random.seed(123)

#chn = np.empty((nst,nwl,ndm),dtype=np.float32)
#stt = np.empty((nst,nwl),dtype=np.float32)

# initialize walkers
xx = np.full((nwl,ndm),x0,dtype=np.float32) + dlt*np.random.randn(nwl,ndm)
#print(xx)
#print(xx.shape[0])
#sstt = np.empty(ndm*nwl,dtype=np.float32)
st = np.empty(nwl,dtype=np.float32)

# initialize statistic
st = np.sum(np.power(xx,2.),axis=1)

chn = np.stack([xx])
stt = np.stack([st])
print(chn)
print(stt)
print(chn.shape)
print(stt.shape)

# generate random walker indices at each step
arrays = [np.random.randint(0, int(nwl/2)-1, size=nwl) for i in range(nst)]
wind = np.stack(arrays,axis=0)
#print(wind)
#print(wind.shape[0])

# generate z random numbers
# distributed as 1/sqrt(z) in the range [1/a,a]
zz = np.random.rand(nst,nwl)
zz = 1./ACONST*np.power(zz*(ACONST-1.)+1,2.)
#print(zz)

# generate r random numbers
rr = np.random.rand(nst,nwl)
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
    xxnp = xxC[inC,:]
    xx1 = xxnp + (xx0 - xxnp)*zz0[:,None]
    st1 = np.sum(np.power(xx1,2.),axis=1)
    qq = np.exp(-0.5 * (st1 - st0)) * np.power(zz0,ndm - 1)
    more = np.array(qq>rr0,dtype=np.int32)
    less = np.array(qq<=rr0,dtype=np.int32)
    xx0 = more[:,None]*xx1 + less[:,None]*xx0
    st0 = more*st1 + less*st0
    ##
    xx = np.append(xx0,xxC,axis=0)
    st = np.append(st0,stC)
    ##
    xx0 = xx[int(nwl/2):,:]
    st0 = st[int(nwl/2):]
    xxC = xx[:int(nwl/2),:]
    stC = st[:int(nwl/2)]
    inC = wind[i,:int(nwl/2)]
    zz0 = zz[i,int(nwl/2):]
    rr0 = rr[i,int(nwl/2):]
    ##
    xxnp = xxC[inC,:]
    xx1 = xxnp + (xx0 - xxnp)*zz0[:,None]
    st1 = np.sum(np.power(xx1,2.),axis=1) #l2norm_kernel(xx1, axis=1)
    qq = np.exp(-0.5 * (st1 - st0)) * np.power(zz0,ndm - 1)
    more = np.array(qq>rr0,dtype=np.int32)
    less = np.array(qq<=rr0,dtype=np.int32)
    xx0 = more[:,None]*xx1 + less[:,None]*xx0
    st0 = more*st1 + less*st0
    ##
    xx = np.append(xxC,xx0,axis=0)
    st = np.append(stC,st0,axis=0)
    ##
    chn = np.append(chn,np.stack([xx]),axis=0)
    stt = np.append(stt,np.stack([st]),axis=0)


stps = linspace(1,nst,num=nst+1)
fig, ax = plt.subplots(nrows=4)
plt.subplots_adjust(hspace=0.1)

for i in range(3):
    for j in range(nwl):
        ax[i].errorbar(stps,chn[:,j,i])
for j in range(nwl):
    ax[3].errorbar(stps,stt[:,j])
#plt.show()
plt.savefig("simplegauss_chain_np"+".png")

#np.random.default_rng.shuffle(wind,axis=1)
#rn = np.random.rand(int(3*nst*nwl),dtype=np.float32)
#print(rn)
