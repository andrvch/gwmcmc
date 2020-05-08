#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from cudakde import *
from matplotlib import animation

smpls = read_data(sys.argv[1])
nwlkrs = int(sys.argv[2])

ns = 2
nd = 2
nt = 20

nprmtrs = shape(smpls)[0]
nstps = int(shape(smpls)[1]/float(nwlkrs))

print nstps, nprmtrs

wlkrs = np.empty([nd,nt,ns,nwlkrs,nstps])

for i in range(nstps):
    for j in range(nwlkrs):
        for s in range(ns):
            for t in range(nt):
                for l in range(nd):
                    wlkrs[l,t,s,j,i] = smpls[l+t*nd+s*nd*nt,j+nwlkrs*i]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0., 1.), ylim=(0., 1.))
line1, = ax.plot([], [], 'o')
line2, = ax.plot([], [], 'o')
line3, = ax.plot([], [], '*')
line4, = ax.plot([], [], '*')
n_text = ax.text(0.70, 0.80, '', transform=ax.transAxes, fontsize=16)
plt.tick_params(labelsize=14)
plt.legend(fontsize=16)
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)

# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    n_text.set_text(r'$n = $')
    return line1, line2,  line3, line4,

# animation function.  This is called sequentially
def animate(i):
    line1.set_data(wlkrs[0,:,0,:5,i],wlkrs[1,:,0,:5,i])
    line2.set_data(wlkrs[0,:,1,:5,i],wlkrs[1,:,1,:5,i])
    line3.set_data(wlkrs[0,0,0,:5,i],wlkrs[1,0,0,:5,i])
    line4.set_data(wlkrs[0,0,1,:5,i],wlkrs[1,0,1,:5,i])
    n_text.set_text(r'$t = %i$' % i )
    return line1, line2, line3, line4,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=128, interval=1, blit=True)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('gauss2D.gif', dpi=80, writer='imagemagick')

#plt.show()
#plt.savefig(sys.argv[1]+".pdf")
