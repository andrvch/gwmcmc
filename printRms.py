#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import random
import time
from cudakde import *

pi = 3.14159265358979323846

def ParseInput(infile):
    f = open(infile)
    lines = []
    for l in f:
        lines.append(str(l))
    coords = []
    source_num_old=0
    for i in range(len(lines)):
        try:
            ls=lines[i].split()
            if ls[0].strip() == 'Source':
                if ls[1].strip(',') != "PSR":
                    source_num = float(ls[1].strip(','))
                else:
                    source_num = 55
            elif ls[0] == 'Offset':
                offx,offy  = tuple(ls)[3:5]
            elif ls[0] == 'Pos':
                errelmaj,errelmin,errelang = tuple(ls)[4:7]
                if source_num != source_num_old:
                    coords.append((float(offx),float(offy),float(errelmaj),float(errelmin),(pi/180.)*float(errelang)))
                    source_num_old = source_num
                else:
                    coords[-1] = tuple(float(offx),float(offy),float(errelmaj),float(errelmin),(pi/180.)*float(errelang))
        except:
            pass
    coords_1 = np.empty([len(coords),5])
    for i in range(len(coords)):
        coords_1[i] = np.array([coords[i]])
    return coords_1

def FoldInput(infile):
    inf = open(infile)
    lines = inf.readlines()
    epochs = np.zeros([len(lines)])
    epochs[0] = lines[0].split()[1]
    coords_array = np.array([ParseInput(lines[0].split()[0])])
    for i in range(len(lines)-1):
        coords_array = np.append(coords_array,np.array([ParseInput(lines[i+1].split()[0])]),axis=0)
        epochs[i+1] = lines[i+1].split()[1]
    return coords_array, epochs


imgs = sys.argv[1]

crds,epchs = FoldInput(imgs)
crdsref = crds[:,:-1,:]
crdspsr = crds[:,-1:,:]

nim = shape(crdsref)[0]
nst = shape(crdsref)[1]
print nim, nst
print epchs
nref = 1

pars = read_data(sys.argv[2])
print pars.shape

th = pars[1,:]

def rms(th):
    ind = []
    for i in range(nim):
        if i != nref:
            ind.append(i)
    trsf = np.empty([nim,5])
    trsf[nref,:] = np.array([(0.,0.,0.,1.,1.)])
    for i in range(nim-1):
        trsf[ind[i],:] = np.array([(th[5*i],th[5*i+1],th[5*i+2],th[5*i+3],th[5*i+4])])
    xs1 = np.empty([nst])
    xs2 = np.empty([nst])
    for i in range(nst):
        xs1[i] = th[5*(nim-1)+2*i]
        xs2[i] = th[5*(nim-1)+2*i+1]
    xs = xs1+1j*xs2
    xref = crdsref[:,:,0]+1j*crdsref[:,:,1]
    prob = np.empty([nim,nst,2])
    for i in range(nim):
        for j in range(nst):
            delx = exp(1j*trsf[i,2])*xs[j]+trsf[i,0]+1j*trsf[i,1]
            delx = delx.real*trsf[i,3]+1j*delx.imag*trsf[i,4] - xref[i,j]
            rotdelx = delx*exp(1j*crdsref[i,j,4])
            prob[i,j,0] = (rotdelx.real/crdsref[i,j,2])**2
            prob[i,j,1] = (rotdelx.imag/crdsref[i,j,3])**2
    return prob

rmsst = rms(th)

f = open(sys.argv[1]+"."+"rms", "w")
for i in range(nim):
    for j in range(nst):
        f.write(" %.15E %.15E\n "%(rmsst[i,j,0],rmsst[i,j,1]))
        print "image %i, star %i:"%(i,j), rmsst[i,j,0],rmsst[i,j,1]
f.close()
