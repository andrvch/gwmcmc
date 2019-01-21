#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyfits
import os, sys
import numpy as np
import asciitable
import asciitable.latex

FileNames = ['acisf11123_repro_evt2.fits','acisf19165_repro_evt2.fits','acisf20876_repro_evt2.fits']

def list_file(FileName):
    f = pyfits.open(FileName)
    ExpTime = round(float(f[1].header['EXPOSURE']),0)
    Inst = str(f[1].header['INSTRUME'])
    Mission = str(f[1].header['TELESCOP'])
    ObsID = str(f[0].header['OBS_ID'])
    DateObs = str(f[0].header['DATE-OBS'])
    MJD = str(f[0].header['MJD_OBS'])
    Observer = str(f[1].header['OBSERVER'])
    return Mission,Inst,ObsID,Observer,DateObs,MJD,ExpTime

prst = []
for i in range(7):
    lst = []
    for FileName in FileNames:
        lst.append(list_file(FileName)[i])
    prst.append(lst)

data = {'Inst': prst[1], 'ObsID': prst[2], 'DateObs': prst[4], 'MJD': prst[5], 'ExpTime': prst[6]}
asciitable.write(data, sys.stdout, Writer = asciitable.Latex, latexdict = {'preamble': r'\begin{center}', 'tablefoot': r'\end{center}', 'tabletype': 'table*'})
