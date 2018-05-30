#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *

pi   = 3.141592654
GeV  = 1.6022E-3          # GeV in ergs
MeV  = 1.6022E-6          # MeV in ergs  
eV   = 1.6022E-12         # eV in ergs
KeV  = 1.6022E-9          # KeV (in ergs)
eee  = 4.8032E-10         # elementary charge, CGS
hh   = 1.054572E-27       # erg*s
h    = 6.626068E-27       # erg*s, 2*pi*hh
c    = 2.997924E10        # cm/s
me   = 0.510998918        # MeV, electron mass
mp   = 938.27203          # MeV proton mass
mn   = 939.56536          # MeV neutron mass
ang  = 1.E-8              # cm in angstrom
kb   = 1.38E-16           # erg/K
sig  = 5.6704E-5          # erg*cm**{-2}*s**{-1}*K**{-4}
arad = 7.56E-15           # erg*cm**{-3}*K**{-4}
km   = 1.E5               # cm
Ms   = 1.989E33           # g
Rs   = 6.96E10            # cm, the Sun radius
Mn   = 1.4*Ms             # a neutron star mass
Rn   = 10*km              # a neutron star radius
pc   = 3.085E18           # cm, parsec
au   = 1.496E13           # cm
muJy = 1.E29              # \muJy
G    = 6.674E-8           # cm**{3}*s**{-1}*g**{-1} 
yr   = 3.E7                # seconds in year
mpp  = (mp*MeV)*pow(c,-2) # proton mass in g

inf = open(sys.argv[1])
lin = inf.readlines()
en = np.array([float(x) for x in lin[1].split()])
nt = len(lin[0].split())
#nt = 1
fx = np.empty([nt,len(en)])
for i in range(nt):
  fx[i,:] = np.array([float(x) for x in lin[2+i].split()])

inf = open(sys.argv[2])
lin = inf.readlines()
en2 = np.array([float(x) for x in lin[1].split()])
nt2 = len(lin[0].split())
fx2 = np.empty([nt2,len(en2)])
for i in range(nt2):
  fx2[i,:] = np.array([float(x) for x in lin[2+i].split()])

def bbodyrad(x,T):
  ee = pow(T,-1)
  xx = x/KeV
  return (1/h**2/c**2)*pow(xx,2.)*pow(exp(x*ee)-1,-1.)

DR = 13*1.E5/3.E22
KK = (13./DR)**2 

def bbodyradX(x,T,K):
  ee = pow(T,-1)
  return K*1.0344E-03*pow(x,2.)*pow(exp(x*ee)-1,-1.)

T = [0.5500000E+01,0.5600000E+01,0.5700000E+01,0.5800000E+01,0.5900000E+01,0.6000000E+01,0.6100000E+01,0.6200000E+01,0.6300000E+01,0.6400000E+01,0.6500000E+01,0.6600000E+01,0.6700000E+01,0.6800000E+01]
TeV = [10**x/11604./1000. for x in T]
ne = linspace(0.001, 10.0, 1000)

redshift = 1.21

ne = ne/redshift
en = en/redshift
en2 = en2/redshift

Rns = 13.11
D = 2.E3

Norm = (1.21*Rns*km/D/pc)**2
hhh = 10**26.178744

for i in range(4,5):
  plt.plot(en,Norm*fx[i,:],color='k')
  
for i in range(5,6):
  plt.plot(en2,Norm*fx2[i,:],color='c')

nui = 10**14.58*h/KeV
nuerri = 10**13.88*h/KeV
nug = 10**14.79*h/KeV
nuerrg = 10**14.29*h/KeV
Fg = 0.02/muJy
Fi = 0.083/muJy
plt.errorbar(nug,Fg,xerr=nuerrg,fmt='k')
plt.errorbar(nug,Fg-0.2*Fg,fmt='k',yerr=0.2*Fg,lolims=True)

plt.errorbar(nui,Fi,xerr=nuerri,fmt='k')
plt.errorbar(nui,Fi-0.2*Fi,fmt='k',yerr=0.2*Fi,lolims=True)
  
#for i in range(5,6):
#  plt.plot(ne,bbodyradX(ne,TeV[i],KK)*ne,color='k')  

TT = [3.9E5,1.E6,1.15E6,1.2E6,8E5] 
TTeV = [x/11604./1000. for x in TT]
col = ['r','g','b','y','m']

for i in range(len(TTeV)):  
  plt.plot(ne,Norm*bbodyradX(ne*redshift,TTeV[i],KK)*ne*redshift/hhh,color=col[i])    

yscale('log',nonposy='clip')
xscale('log')
plt.show()