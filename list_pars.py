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
        pars.append(pp)
    prst = []
    for j in range(10):
        lst = []
        for i in range(len(lines)):
            lst.append(r'$%1.2f^{+%1.2f}_{-%1.2f}$'%(pars[i][1,j],pars[i][2,j]-pars[i][1,j],pars[i][1,j]-pars[i][0,j]))
        prst.append(lst)
    return prst

prst = FoldInput(sys.argv[1])

models = ['bb','nsa12','nsa13','ns1260','ns123100','ns123190','ns130100','ns130190']

data = {r'A$Mod.$': models, r'B$N_{\rm H}$': prst[6], r'F$\Gamma_{\rm psr}$': prst[2], r'G$K_{\rm psr}$': prst[3], r'C$T$': prst[0], r'D$R$': prst[1], r'E$D$': prst[7], r'H$\Gamma_{\rm pwn}$': prst[4], r'J$K_{\rm pwn}$': prst[5], r'K$W$': prst[8], r'K$\chi^{2}$': prst[9]}
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*', 'units':{'$N_{\rm H}$':'$\rm 10^{21} ./ cm^{-2}$'}})
