#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pylab import *
from astropy.io import ascii
import numpy as np
import emcee
import scipy.optimize as op
from multiprocessing import Pool

lmin = 4.
lmax = 10.

ymin = 1.e-4

nl = 20
ny = 1000

def firstIntegral(y,l):
    x = np.sqrt(l**2 + 1 - y**2 + 2*np.log(y))
    return x

def circle1(y,l):
    x = np.sqrt(l**2 + 1 - y**2)
    return x

def circle2(y,l):
    x = np.sqrt(l**2 - 2.*(y-1.)**2)
    return x

ly1 = math.log10(1. + 1.e-11)
ly2 = 0.9
#ll = np.linspace(lmin,lmax,num=nl)
lymax = np.linspace(ly1,ly2,num=nl)
#ymax = np.linspace(10**ly1,10**ly2,num=nl)

for ly in lymax:
    #ymax = math.sqrt(l**2+1)
    y = 10**ly
    l = math.sqrt(y**2-1.-2*log(y))
    print(l)
    yy = np.linspace(ymin,y,num=ny)
    x = firstIntegral(yy,l)
    #if y > 3.:
    c1 = circle1(yy,l)
    #else:
    c2 = circle2(yy,l)
    plt.plot(yy,x,'k')
    #plt.plot(yy,c1,'g')
    plt.plot(yy,c2,'y--')

plt.savefig("spirals"+".png")
