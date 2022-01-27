#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pylab import *
import numpy as np

pipi = 3.14159265359

N = 1000

omg = 0.1
E0 = 1.
W = 64.
l2 = 1.
l3 = 1.

bt = 2*l2*omg
bt3 = 3*l3*omg

print(bt)
print(bt3)

#lmbd = math.sqrt(l2*l1*(1+bt**2)/(1+bt3**2))
#lmbd3 = 64.*l3/55./pipi/(1+bt3**2)

yy = np.linspace(-W/2.,W/2.,N)

def velocity(y,E0,W,l2,l3,bt,bt3):
    lmbd = math.sqrt(l2*l3*(1+bt**2)/(1+bt3**2))
    lmbd3 = 64.*l3/55./pipi/(1+bt3**2)
    #print(lmbd)
    #print(lmbd3)
    A = bt*E0*W/2./(1+lmbd3/lmbd)*math.exp(-W/2./lmbd)
    #print(A)
    D = E0*W**2/8./l2 + 3*pipi*E0*W/16. + (bt3-bt)*E0*l3*bt*W/2./(1+bt3**2)/(lmbd+lmbd3)
    #print(D)
    v = - E0*y**2/2./l2 + D + l3*(bt+bt3)/lmbd/(1+bt3**2)*2*A*math.cosh(y/lmbd)
    #print(v)
    return v

def velocity_poi(y,E0,W,l2,l3,bt,bt3):
    lmbd = math.sqrt(l2*l3*(1+bt**2)/(1+bt3**2))
    lmbd3 = 64.*l3/55./pipi/(1+bt3**2)
    #print(lmbd)
    #print(lmbd3)
    A = bt*E0*W/2./(1+lmbd3/lmbd)*math.exp(-W/2./lmbd)
    #print(A)
    D = E0*W**2/8./l2 #+ 3*pipi*E0*W/16. + (bt3-bt)*E0*l3*bt*W/2./(1+bt3**2)/(lmbd+lmbd3)
    #print(D)
    v = - E0*y**2/2./l2 + D #+ l3*(bt+bt3)/lmbd/(1+bt3**2)*2*A*math.cosh(y/lmbd)
    #print(v)
    return v

def ehall(y,omg,E0,W,l2,l3,bt,bt3):
    lmbd = math.sqrt(l2*l3*(1+bt**2)/(1+bt3**2))
    lmbd3 = 64.*l3/55./pipi/(1+bt3**2)
    A = bt*E0*W/2./(1+lmbd3/lmbd)*math.exp(-W/2./lmbd)
    pixxslash = - bt*E0 + 2*A/lmbd*math.cosh(y/lmbd)
    vx = velocity(y,E0,W,l2,l3,bt,bt3)
    eh = - omg*vx - pixxslash
    return - eh

def ehall_poi(y,omg,E0,W,l2,l3,bt,bt3):
    lmbd = math.sqrt(l2*l3*(1+bt**2)/(1+bt3**2))
    lmbd3 = 64.*l3/55./pipi/(1+bt3**2)
    A = bt*E0*W/2./(1+lmbd3/lmbd)*math.exp(-W/2./lmbd)
    pixxslash = - bt*E0 #+ 2*A/lmbd*math.cosh(y/lmbd)
    vx = velocity_poi(y,E0,W,l2,l3,bt,bt3)
    eh = - omg*vx - pixxslash
    return - eh

vv = np.empty([N])
vvp = np.empty([N])
eeh = np.empty([N])
eehp = np.empty([N])

for i in range(N):
    vv[i] = velocity(yy[i],E0,W,l2,l3,bt,bt3)
    vvp[i] = velocity_poi(yy[i],E0,W,l2,l3,bt,bt3)
    eeh[i] = ehall(yy[i],omg,E0,W,l2,l3,bt,bt3)
    eehp[i] = ehall_poi(yy[i],omg,E0,W,l2,l3,bt,bt3)
    #print(vv[i])

fig, ax = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.05)

ax[0].text(22, 520, r'$\omega_c = 0.1$', fontsize=11)
ax[0].text(22, 450, r'$W = 64$', fontsize=11)
ax[0].text(22, 390, r'$l_2 = l_3 = 1$', fontsize=11)

ax[0].plot(yy,vv,'-')
ax[0].plot(yy,vvp,'--') #dddddddddddddddddddddddddddsssßsssssssßßßßßßßsssssßßßßßßssssssßßsssssßssßs
ax[0].grid(color = 'grey', linestyle = '--', linewidth = 0.5)
ax[0].set_ylim(0, np.max(vv)+0.05*np.max(vv))
ax[0].set_xlim(-W/2., W/2.)
ax[0].set_ylabel(r"$v_x$",fontsize=12)
setp([a.get_xticklabels() for a in ax[:1]], visible=False)

ax[1].text(22, 52, r'$\omega_c = 0.1$', fontsize=11)
ax[1].text(22, 45, r'$W = 64$', fontsize=11)
ax[1].text(22, 39, r'$l_2 = l_3 = 1$', fontsize=11)
ax[1].grid(color = 'grey', linestyle = '--', linewidth = 0.5)
ax[1].set_xlim(-W/2., W/2.)
ax[1].set_ylabel(r"$E_H$",fontsize=12)
ax[1].set_xlabel(r"$y$",fontsize=12)
ax[1].set_ylim(0, np.max(eeh)+0.05*np.max(eeh))
ax[1].plot(yy,eeh,'-')
ax[1].plot(yy,eehp,'--')


plt.savefig("fig3_1"+".pdf")
