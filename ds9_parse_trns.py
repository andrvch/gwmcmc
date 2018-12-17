#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
from pylab import *
import numpy as np
import emcee
#from scipy.optimize import curve_fit
#from scipy.optimize import minimize, show_options
import scipy.optimize as op
from astropy import units as u
from astropy.coordinates import SkyCoord

pi   = 3.141592654

Cords1       = sys.argv[1]
Cords2       = sys.argv[2]


def parse_input(in_file):
  f = open(in_file)
  lines = []
  for l in f:
    lines.append(str(l))
  coords = []
  for i in range(len(lines)):
    try:
      #print lines[i].split()
      if lines[i].split()[0] == 'Offset':
        offx,offy  = tuple(lines[i].split())[3:5]
      elif  lines[i].split()[0] == 'Pos':
        errelmaj,errelmin,errelang = tuple(lines[i].split())[4:7]
      elif lines[i].split()[0] == 'Right':
        ra = lines[i].split()[2]
      elif lines[i].split()[0] == 'Declination:':
        dec = lines[i].split()[1]
        cc = SkyCoord(ra+' '+dec, frame='icrs', unit=(u.hourangle, u.deg))
        coords.append((float(offx),float(offy),float(errelmaj),float(errelmin),float(errelang),float(cc.ra.degree),float(cc.dec.degree)))
    except:
      pass
    coords_1 = np.empty([len(coords),7])
    for i in range(len(coords)):
      coords_1[i] = np.array([coords[i]])  
  return coords_1

coords1 = parse_input(Cords1)
coords2 = parse_input(Cords2)
psr1 = coords1[-1:,:] #parse_input(psr_coords1)
psr2 = coords2[-1:,:] #parse_input(psr_coords2)
coords1 = coords1[:-1,:]
coords2 = coords2[:-1,:]

#print len(coords1)
print 'CODRS'
#print coords1[:,5]
#print coords1[:,0]
ref_delta1 = psr1[:,6] - (1/3600.)*psr1[:,1]
cosdelta1  = np.cos((pi/180.)*ref_delta1)
ref_alpha1 = psr1[:,5] - (1/3600.)*psr1[:,0]/cosdelta1
print ref_delta1
print ref_alpha1
print cosdelta1
print psr1[:,5:]

exit()
off_psr_alpha = 1.09791129
off_psr_delta = -2.52083887
#print cosdelta1
#print ref_alpha1 #+ (1/3600.)*0.174709/cosdelta1
#print coords1[:,6]
#print coords1[:,1]
#print ref_delta1 #+ (1/3600.)*(-29.326)
#print coords1[:,5:]
#exit()

z0  = -0.02221384+1j*(-1.04942216)
phi = -1.75641575e-05
#print z0
#print phi

s1 = 1.00212619
s2 = 1.00199071

def transform_coords(z0,phi,s1,s2,coords_ref,coords_tar):
  ref_delta = coords_ref[0,6] - (1/3600.)*coords_ref[0,1]
  cosdelta  = np.cos((pi/180.)*ref_delta)
  ref_alpha = coords_ref[0,5] - (1/3600.)*coords_ref[0,0]/cosdelta
  x1 = (1./s1)*coords_tar[:,0]+1j*(1./s2)*coords_tar[:,1]
  x2 = exp(-1j*phi)*(x1-z0)
  coords = coords_tar
  coords[:,5] = ref_alpha + (1/3600.)*(x2.real/cosdelta)
  coords[:,6] = ref_delta + (1/3600.)*(x2.imag)
  return coords

coords     = transform_coords(z0,phi,s1,s2,coords1,coords2)
#print coords
coords_psr = transform_coords(z0,phi,s1,s2,psr1,psr2) 
print psr1[:,2:]
print coords_psr[:,2:]

print (coords_psr[0,6] - psr1[0,6])*3600
exit()

namereg = Cords2 + 'TR'  + '.reg'

f = open(namereg, "w")  

f.write('global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=1 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n') #% (alpha[i],delta[i],ref[i]))

n = len(coords)

#for i in range(n):        
#    f.write('ellipse(%5.7f,%5.7f,%5.5f",%5.5f",%5.5f) # text={%-2i}\n'% (coords[i,5],coords[i,6],coords[i,2],coords[i,3],coords[i,4]+90,i+1))
i=0
f.write('ellipse(%5.7f,%5.7f,%5.5f",%5.5f",%5.5f) # text={%-2i}\n'% (coords[i,5],coords[i,6],coords[i,2],coords[i,3],coords[i,4]+90,0))    

f.close()
