#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
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
    pi_text.set_text(r'$\pi = $')
    n_text.set_text(r'$n = $')
    #line2.set_data([], [])
    return line1, #line2,

# animation function.  This is called sequentially
def animate(i):
    x = np.random.rand(2*i+10)
    y = np.random.rand(2*i+10)
    for j in range(len(x)):
        if (x[j]**2+y[j]**2<=1):
            xx.append(x[j])
            yy.append(y[j])
        else:
            xx1.append(x[j])
            yy1.append(y[j])
    m =+ len(xx) + len(xx1)
    n =+ len(xx)
    pp = float(n) / float(m)
    print n, m, pp*4
    line1.set_data(np.array(xx),np.array(yy))
    pi_text.set_text(r'$\pi = %1.4f$' % (pp*4))
    n_text.set_text(r'$n = %i$' % m )
    #line2.set_data(x2,y2)
    return line1, #line2,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=20, blit=True)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('basic.gif', dpi=80, writer='imagemagick')

#plt.savefig("sphere.pdf")

#plt.show()
