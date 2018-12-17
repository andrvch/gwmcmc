#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
from pylab import *
from astropy.io import ascii
import numpy as np
import emcee
import scipy.optimize as op
from StringIO import StringIO

from astropy import units as u
from astropy.coordinates import SkyCoord


pi = 3.14159265358979323846

ref_delta1 = -47.74291667 # #coords1[0,6] - (1/3600.)*coords1[0,1]
cosdelta1  = np.cos((pi/180.)*ref_delta1)
ref_alpha1 = 262.92541638 # #coords1[0,5] - (1/3600.)*coords1[0,0]/cosdelta1

ref_delta2 = -47.74277778
cosdelta2  = np.cos((pi/180.)*ref_delta2)
ref_alpha2 = 262.92542947


#In_file_2011 = sys.argv[1]

ra_1 = '17:31:42.263'#17:31:42.123
dec_1 = '-47:44:38.28' #-47:44:36.20'
ra_2 = '17:31:42.215'
dec_2 = '-47:44:37.05'
cc_1 = SkyCoord(ra_1+' '+dec_1, frame='icrs', unit=(u.hourangle, u.deg))
cc_2 = SkyCoord(ra_2+' '+dec_2, frame='icrs', unit=(u.hourangle, u.deg))

print (float(cc_1.ra.degree) - float(cc_2.ra.degree))*3600/5.*cosdelta1
print (float(cc_1.dec.degree) - float(cc_2.dec.degree))*3600/5.

exit()
def parse_input(in_file):
    f = open(in_file)
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
            elif lines[i].split()[0] == 'Right':
                ra = lines[i].split()[2]
            elif lines[i].split()[0] == 'Declination:':
                dec = lines[i].split()[1]
                cc = SkyCoord(ra+' '+dec, frame='icrs', unit=(u.hourangle, u.deg))
                if source_num != source_num_old:
                    coords.append((float(offx),float(offy),float(errelmaj),float(errelmin),(pi/180.)*float(errelang),float(cc.ra.degree),float(cc.dec.degree)))
                    source_num_old = source_num
                else:
                    coords[-1] = tuple(float(offx),float(offy),float(errelmaj),float(errelmin),(pi/180.)*float(errelang),float(cc.ra.degree),float(cc.dec.degree))   
        except:
            pass
    coords_1 = np.empty([len(coords),7])
    for i in range(len(coords)):
        coords_1[i] = np.array([coords[i]])  
    return coords_1


coords_2011 = parse_input(In_file_2011)

#exit()
for i in range(shape(coords_2011)[0]):
    print coords_2011[i,0], coords_2011[i,1], (ref_alpha1-coords_2011[i,5])*3600*cosdelta1, (ref_delta1-coords_2011[i,6])*3600
    

