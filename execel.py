#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import pyexcel
import pandas as pd

sheet1 = pyexcel.get_sheet(file_name="Untitled5.xlsx")#, name_columns_by_row=0)
sheet2 = pyexcel.get_sheet(file_name="Untitled4.xlsx")#, name_columns_by_row=0)

sheet1.name_columns_by_row(0)
print(sheet1)
sheet2.name_columns_by_row(0)
print(sheet2)

print(sheet1.column['file'])

exe1 = [sheet1.column['size, Mb'][i] for i in [i for i, j in enumerate(sheet1.column['type']) if j == 'sound']]
indx = [i for i, j in enumerate(sheet1.column['type']) if j == 'sound']
print(indx)
print(exe1)
print(sum(exe1)/len(exe1))


filetype1 = list(zip(sheet1.column['file'],sheet1.column['type'],sheet1.column['last modification date']))

file2 = [sheet2.column['file'][i] for i in [i for i, j in enumerate(sheet2.column['priority']) if j == 'low']]
type2 = [sheet2.column['type'][i] for i in [i for i, j in enumerate(sheet2.column['priority']) if j == 'low']]

filetype2 = list(zip(file2,type2))

exe2 = [k for i,j,k in filetype1 if (i,j) in filetype2]
print(exe2)
print(min(exe2))

ft2 = list(zip(sheet2.column['file'],sheet2.column['type']))

ft1 = list(zip(sheet1.column['file'],sheet1.column['type'],sheet1.column['size, Mb']))

ex3 = [(i,j,k) for i,j,k in ft1 if (i,j) not in ft2]
print(ex3)



exit()

dict1 = sheet1.to_dict()
dict2 = sheet2.to_dict()
print(dict1)
print(dict2)
print(type(dict1['file']))

if 'sea' in dict1['file']:
    print('that is fine')
