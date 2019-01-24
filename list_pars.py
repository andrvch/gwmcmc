#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyfits
import os, sys
import numpy as np
import asciitable
import asciitable.latex
from cudakde import *

def FoldInput(infile):
    inf = open(infile)
    lines = inf.readlines()
    pars = []
    for line in lines:
        pp = read_data(line.split()[0])
        pp[1,1] = 10**pp[1,1]*13.
        pp[0,1] = 10**pp[0,1]*13.
        pp[2,1] = 10**pp[2,1]*13.
        pp[1,3] = 10**pp[1,3]/1.e-5
        pp[0,3] = 10**pp[0,3]/1.e-5
        pp[2,3] = 10**pp[2,3]/1.e-5
        pp[1,5] = 10**pp[1,5]/1.e-5
        pp[0,5] = 10**pp[0,5]/1.e-5
        pp[2,5] = 10**pp[2,5]/1.e-5
        pp[1,13] = 10**pp[1,13]
        pp[0,13] = 10**pp[0,13]
        pp[2,13] = 10**pp[2,13]
        pars.append(pp)
    prst = []
    for j in range(14):
        lst = []
        for i in range(len(lines)):
            lst.append(r'$%1.2f^{+%1.2f}_{-%1.2f}$'%(pars[i][1,j],pars[i][2,j]-pars[i][1,j],pars[i][1,j]-pars[i][0,j]))
        prst.append(lst)
    return prst

prst = FoldInput(sys.argv[1])

models = ['123190']

data = {r'A$Mod.$': models, r'B$N_{\rm H}$': prst[12], r'C$\Gamma_{\rm psr}$': prst[2], r'D$K_{\rm psr}$': prst[3], r'E$T$': prst[0], r'F$R$': prst[1], r'G$D$': prst[13], r'H$\Gamma_{\rm pwn}$': prst[4], r'J$K_{\rm pwn}$': prst[5]}
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*', 'units':{'$N_{\rm H}$':'$\rm 10^{21} ./ cm^{-2}$'}})
