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
    prst = []
    for j in range(7):
        lst = []
        for line in lines:
            lst.append(read_data(line.split()[0])[j])
        prst.append(lst)
    return prst

prst = FoldInput(sys.argv[1])

data = {'Model': prst[1], 'ObsID': prst[2], 'DateObs': prst[4], 'MJD': prst[5], 'ExpTime': prst[6]}
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*'})
