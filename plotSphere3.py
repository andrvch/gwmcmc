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

nprmtrs = shape(smpls)[0]
nstps = int(shape(smpls)[1]/float(nwlkrs))

print nstps, nprmtrs

wlkrs = np.empty([nprmtrs,nwlkrs,nstps])

for i in range(nstps):
    for j in range(nwlkrs):
        for k in range(nprmtrs):
            wlkrs[k,j,i] = smpls[k,j+nwlkrs*i]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-2., 2.), ylim=(-2., 2.))
line1, = ax.plot([], [], 'o')
line2, = ax.plot([], [], 'o')
pi_text = ax.text(0.70, 0.90, '', transform=ax.transAxes, fontsize=16)
n_text = ax.text(0.70, 0.80, '', transform=ax.transAxes, fontsize=16)
plt.tick_params(labelsize=14)
plt.legend(fontsize=16)
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)

xx = []
yy = []
xx1 = []
yy1 = []

m = 0
n = 0
pp = 0

#x = np.random.rand(200000)
#y = np.random.rand(200000)

# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    #pi_text.set_text(r'$\pi = $')
    #n_text.set_text(r'$n = $')
    #line2.set_data([], [])
    return line1, #line2,

# animation function.  This is called sequentially
def animate(i):
    line1.set_data(wlkrs[0,:,i],wlkrs[1,:,i])
    #pi_text.set_text(r'$\pi = %1.4f$' % (pp*4))
    n_text.set_text(r'$t = %i$' % i )
    #line2.set_data(x2,y2)
    return line1, #line2,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=512, interval=32, blit=True)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('gauss2D.gif', dpi=80, writer='imagemagick')

#plt.show()
#plt.savefig(sys.argv[1]+".pdf")
