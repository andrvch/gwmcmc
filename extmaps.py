#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, sys
import numpy as np
import matplotlib.pyplot as plt
import mwdust
from pylab import *

#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams['text.usetex'] = True

#marshall = mwdust.Marshall06(sf10=True)
drimmel  = mwdust.Drimmel03(sf10=True)
#green    = mwdust.Green15(sf10=True)
#sale     = mwdust.Sale14(sf10=True)
zero     = mwdust.Zero(sf10=True)
#sfd      = mwdust.SFD(sf10=True)
#combined = mwdust.Combined15(sf10=True)

#D        = np.array([0.25,0.5,1.,2.,3.,4.,5.,6.])
D        = np.linspace(0.1,15.,1000)
L        = 342.57 # 54.7
B        = -7.67 #0.08

#plt.plot(D,marshall(L,B,D),'k-',label=r'$\rm Marshall \, et \, al. \, (2006)$',linewidth=2.0)
plt.plot(D,drimmel(L,B,D),'y-',label=r'$\rm Drimmel \, et \, al. \, (2003)$',linewidth=2.0)
#plt.plot(D,green(L,B,D),'g-',label=r'$\rm Green \, et \, al. \, (2015)$',linewidth=2.0)
#plt.plot(D,sale(L,B,D),'b-',label=r'$\rm Sale \, et \, al. \, (2014)$',linewidth=2.0)
#plt.plot(D,zero(L,B,D),'k--',label='-)')
#plt.plot(np.array([0.1,20.]),sfd(L,B,np.array([0.1,1.])),'c--',label='Schlegel et al. (1998)')
#plt.plot(D,combined(L,B,D),'k-',linewidth=2.0,label='Bovy et al. (2015)')

#plt.plot([0.1,20.],[1.7,1.7],'k--',linewidth=2.0)
plt.plot([0.1,20.],[0.12,0.12],'k--',linewidth=2.0)
#plt.plot([2.2466,2.2466],[0.,1.3],'k:',linewidth=2.0)
#plt.plot([6.02976,6.02976],[0.,2.1],'k:',linewidth=2.0)
#plt.plot([2.2466,6.02976],[0.001,0.001],'k-',linewidth=3.0)
#plt.fill(np.array([0.1,20.,20.,0.1]),np.array([1.3,1.3,2.1,2.1]),linewidth=0.001,color='0.5')

plt.legend(loc='best')
plt.setp(plt.gca().get_legend().get_texts(), fontsize=18)

plt.xlabel(r'$\rm Distance \, [ \, kpc \, ]$',fontsize=18)
plt.ylabel(r'$E(B-V)$',fontsize=18)
plt.tick_params(labelsize=16)
plt.minorticks_on()
plt.show()

#print hp.maptype(plmap)
