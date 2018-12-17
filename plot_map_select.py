#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, sys
import numpy as np
from scipy import *
from astropy.io import ascii
from scipy.optimize import curve_fit
from scipy.optimize import minimize, show_options
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
import kdist
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

pc = 3.086E18
pi = 3.14159

FileName  = sys.argv[1]

L     = float(sys.argv[2]) #342.57 
B     = float(sys.argv[3]) #-7.67 

vel1  = float(sys.argv[4])
vel2  = float(sys.argv[5])
vel10 = -8.
vel20 = 1.

hh = 1.822E18 # Metthews et al. (1997), eq[8]

hdulist = fits.open(FileName)
prihdr  = hdulist[0].header
velref  = prihdr['CRVAL3'] #-1.885000E+01
nveldel = prihdr['CRPIX3']
delvel  = prihdr['CDELT3'] #1.3
Lref    = prihdr['CRVAL1']
Ldelta  = prihdr['CDELT1']
nlref   = prihdr['CRPIX1']
Bref    = prihdr['CRVAL2']
Bdelta  = prihdr['CDELT2']
nbref   = prihdr['CRPIX2']

def hi_vel(hduName,L,B,vel1,vel2):
    nl      = int(nlref) + int(round((L - Lref)/Ldelta))
    nb      = int(nbref) + int(round((B - Bref)/Bdelta))
    print nl,nb
    data    = hdulist[0].data[:,nb,nl]
    vlsr    = 1.E-3*np.linspace(velref,velref+shape(data)[0]*delvel,shape(data)[0])
    nhI = np.empty([shape(data)[0]])
    for i in range(shape(data)[0]):
        nhI[i] = 1.E-3*delvel*hh*data[:i].sum()
    sel = (vlsr<vel2)*(vlsr>vel1)  
    nhIvoid    = 1.E-3*delvel*hh*data[np.where(sel)]
    return data,vlsr,nhI,nhIvoid

def himap(hduName,L,B,vel1,vel2):
    nl      = int(nlref) + int(round((L - Lref)/Ldelta))
    nb      = int(nbref) + int(round((B - Bref)/Bdelta))
    mapm    = hdulist[0].data[:,:,:]
    vlsr    = 1.E-3*np.linspace(velref,velref+shape(mapm)[0]*delvel,shape(mapm)[0])
    sel     = (vlsr<vel2)*(vlsr>vel1)  
    nhImap  = 1.E-3*delvel*hh*mapm[np.where(sel),:,:]
    #nhImap    = mapm[np.where(sel),:,:]
    return nhImap[0], vlsr[np.where(sel)], [nl,nb]

def inside_poli(x,y,poli):
    n = len(poli)
    inside = False
    j = n - 1
    for i in range(n):
        if ((poli[i,1]<y) and (poli[j,1]>=y)) or ((poli[j,1]<y) and (poli[i,1]>=y)):
            if poli[i,0]+(y-poli[i,1])/(poli[j,1]-poli[i,1])*(poli[j,0]-poli[i,0])<x:
                inside = not inside
        j=i
    return inside    

def select_background_3poli(nhmap,polygon1,polygon2,polygon3):
    smap    = np.array([nhmap[:,int(round(polygon1[0,1])),int(round(polygon1[0,0]))]])
    for j in range(shape(nhmap)[1]):
        for k in range(shape(nhmap)[2]): 
            if inside_poli(k,j,polygon2) or inside_poli(k,j,polygon3):
                smap = np.append(smap,np.array([nhmap[:,j,k]]),axis=0)
    return smap

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xys = np.array([])# list(line.get_xdata())
#        self.ys = []#list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        #print 'click', event
        if (event.button == 3) and (len(self.xys)>0):
            self.xys=self.xys[0:-1]
            self.line.set_data(self.xys[:,0], self.xys[:,1])
            self.line.figure.canvas.draw()
            return
            
        if event.inaxes!=self.line.axes: return
        if len(self.xys)==0:
            self.xys=np.array([[event.xdata,event.ydata]])
        else:
            self.xys=np.append(self.xys,[[event.xdata,event.ydata]],axis=0)
#        self.ys.append(event.ydata)
#        print self.xys
        self.line.set_data(self.xys[:,0], self.xys[:,1])
        self.line.figure.canvas.draw()
        

nhImap, vlsr, cent = himap(hdulist,L,B,vel1,vel2)        

prihdr['CUNIT3'] = 'm/s'
prihdr['NAXIS']  = 2
wcs = WCS(prihdr)

nhImap0 = nhImap[np.where((vlsr<vel20)*(vlsr>vel10)),:,:]


fig  = plt.figure()
ax   = plt.subplot(111) #,projection=wcs)
cax  = ax.imshow(np.sum(nhImap0[0],axis=0), clim=(0.15E21, 1.25E21), origin='lower', cmap='gist_earth')
fig.colorbar(cax)
plt.title('Draw a polygon: \n (Use right and left mouse buttons)')
line, = ax.plot([cent[0]], [cent[1]],'o-')  # empty line

linebuilder = LineBuilder(line)

plt.show()

polygon1 = linebuilder.xys

print polygon1
print shape(polygon1)


fig  = plt.figure()
ax   = plt.subplot(111) #,projection=wcs)
cax  = ax.imshow(np.sum(nhImap0[0],axis=0), clim=(0.15E21, 1.25E21), origin='lower', cmap='gist_earth')
fig.colorbar(cax)

plt.title('Draw a polygon: \n (Use right and left mouse buttons)')
line, = ax.plot([cent[0]], [cent[1]],'o-')  # empty line

linebuilder = LineBuilder(line)

plt.show()

polygon2 = linebuilder.xys
print polygon2


fig = plt.figure()
ax  = plt.subplot(111) #,projection=wcs)
cax  = ax.imshow(np.sum(nhImap0[0],axis=0), clim=(0.15E21, 1.25E21), origin='lower', cmap='gist_earth')
fig.colorbar(cax)
plt.title('Draw a polygon: \n (Use right and left mouse buttons)')
line, = ax.plot([cent[0]], [cent[1]],'o-')  # empty line

linebuilder = LineBuilder(line)

plt.show()

polygon3 = linebuilder.xys
print polygon3

#nl      = int(nlref) + int(round((L - Lref)/Ldelta))
#nb      = int(nbref) + int(round((B - Bref)/Bdelta))

nh_1 = select_background_3poli(nhImap,polygon1,polygon2,polygon3)
print nh_1
print shape(nh_1)

vel_hole = [-10.,5.]
sel = (vlsr<vel_hole[1])*(vlsr>vel_hole[0])
nh_hole = abs((nh_1[0]-np.average(nh_1[1:,:],axis=0))[np.where(sel)].sum())
print nh_hole

vlsrvlsr = linspace(vel_hole[0], vel_hole[1], 100)
distdist = np.array([kdist.kdist(L,B,v) for v in vlsrvlsr])
print distdist
ll       = abs(distdist[-1] - distdist[0])
llcm     = ll*pc
print ll

ang_dim = 4. # deg
Dist    = 800.
ll_2deg_cm = pi*(ang_dim/180.)*Dist*pc
print nh_hole/ll_2deg_cm
print kdist.kdist(L,B,-3.)


gs = gridspec.GridSpec(2,1)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0],sharex=ax1)
#ax3 = plt.subplot(gs[2,0],sharex=ax1)

#ax1.errorbar(distdist,vlsrvlsr,color='g',fmt='-',capsize=0)
ax1.errorbar(vlsr,nh_1[0],color='g',fmt='-',capsize=0)
ax1.errorbar(vlsr,np.average(nh_1[1:,:],axis=0),color='k',fmt='--',capsize=0)
ax2.errorbar(vlsr,nh_1[0]-np.average(nh_1[1:,:],axis=0),color='g',fmt='-',capsize=0)
ax2.errorbar([vlsr[0],vlsr[-1]],[0.,0.],color='k',fmt='--',capsize=0)


plt.show()





exit()



TB,vlsr,nhI,nhvoid = hi_vel(FileName,L,B,vel1,vel2)

print shape(nhvoid)
print nhvoid.sum()

#dist100  = kdist.kdist(L, B, 50)
#print dist100
vlsrvlsr = linspace(vel1, vel2, 100)
distdist = np.array([kdist.kdist(L,B,v) for v in vlsrvlsr])
print distdist
ll       = abs(distdist[-1] - distdist[0])
llcm     = ll*pc
print ll
print nhvoid.sum()/llcm

nhImap = himap(FileName,vel1,vel2)

print shape(nhImap)
print shape(nhImap[0])

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


gs = gridspec.GridSpec(2,1)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0],sharex=ax1)

#ax1.errorbar(distdist,vlsrvlsr,color='g',fmt='-',capsize=0)
ax1.errorbar(vlsr,TB,color='g',fmt='o',capsize=0)
ax2.errorbar(vlsr,nhI,color='g',fmt='o',capsize=0)

plt.show()
