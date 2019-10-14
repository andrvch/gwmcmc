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
    for j in range(7):
        lst = []
        for i in range(len(lines)):
            lst.append(r'$%1.2f^{+%1.2f}_{-%1.2f}$'%(pars[i][1,j],pars[i][2,j]-pars[i][1,j],pars[i][1,j]-pars[i][0,j]))
        prst.append(lst)
    return prst

prst = FoldInput(sys.argv[1])

models = ['bb','nsa12','nsa13','ns1260','ns123100','ns123190','ns130100','ns130190']

data = {r'A $Mod.$': models, r'B Bol. lum.': prst[0], r'C $Psr. flux$': prst[1], r'D PWN flux': prst[2], r'E Psr. Lum.': prst[3], r'F PWN Lum': prst[4], r'G Psr. eff.': prst[5], r'H PWN eff': prst[6]}
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*')
