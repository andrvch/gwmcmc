#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mwdust
from pylab import *

L = 205.09
B = -0.93

ifscale = True

#marshall = mwdust.Marshall06(sf10=True)
drimmel = mwdust.Drimmel03(sf10=ifscale)
green = mwdust.Green15(sf10=ifscale)
sale = mwdust.Sale14(sf10=ifscale)
#zero = mwdust.Zero(sf10=ifscale)
sfd = mwdust.SFD(sf10=ifscale)
combined = mwdust.Combined15(sf10=ifscale)

Drange = [0.09, 21.]
Ndist = 1000
D = np.linspace(Drange[0],Drange[1],Ndist)

f = open("Green15.dat", "w")
for i in range(Ndist):
    f.write("%.15E %.15E\n"%(D[i],green(L,B,D)[i]))
f.close()

#plt.plot(D,marshall(L,B,D),'k-',label=r'$\rm Marshall \, et \, al. \, (2006)$',linewidth=2.0)
plt.plot(D,drimmel(L,B,D),'y-',label=r'$\rm Drimmel \, et \, al. \, (2003)$',linewidth=2.0)
plt.plot(D,green(L,B,D),'g-',label=r'$\rm Green \, et \, al. \, (2015)$',linewidth=2.0)
plt.plot(D,sale(L,B,D),'b-',label=r'$\rm Sale \, et \, al. \, (2014)$',linewidth=2.0)
#plt.plot(D,zero(L,B,D),'k--',label='-)')
plt.plot(np.array([Drange[0],Drange[1]]),sfd(L,B,np.array([Drange[0],Drange[1]])),'c--',label='Schlegel et al. (1998)')
#plt.plot(D,combined(L,B,D),'k-',linewidth=2.0,label='Bovy et al. (2015)')

#plt.plot([0.1,20.],[1.7,1.7],'k--',linewidth=2.0)
#plt.plot([0.1,20.],[0.12,0.12],'k--',linewidth=2.0)
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

plt.savefig('extmaps.eps')
#plt.show()
