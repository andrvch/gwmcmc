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

f1 = sheet1.column['file']
t1 = sheet1.column['type']
s1 = sheet1.column['size, Mb']
f2 = sheet2.column['file']
t2 = sheet2.column['type']
fts1 = list(zip(f1,t1,s1))
ft2 = list(zip(f2,t2))



ft2 = list(zip(sheet2.column['file'],sheet2.column['type']))

ft1 = list(zip(sheet1.column['file'],sheet1.column['type'],sheet1.column['size, Mb']))

ex3 = [(i,j,k) for i,j,k in ft1 if (i,j) not in ft2]
print(ex3)

x = [d for f,t,d in ftd1 if (f,t) in [(f,t) for f,t,p in ftp2 if p=='low']]



exit()

dict1 = sheet1.to_dict()
dict2 = sheet2.to_dict()
print(dict1)
print(dict2)
print(type(dict1['file']))

if 'sea' in dict1['file']:
    print('that is fine')


# Import data from files "Untitled1.xlsx" and "Untitled2.xlsx":
sheet1 = pyexcel.get_sheet(file_name="Untitled1.xlsx")
sheet2 = pyexcel.get_sheet(file_name="Untitled2.xlsx")
sheet1.name_columns_by_row(0)
sheet2.name_columns_by_row(0)
# Get columns 'file', 'type' and 'size, Mb'
# from table 1 and combine them to list of triples
# ('file', 'type', 'size'). Get columns 'file' and 'type' from
# table 2 and then combine them to list of pairs
# ('file','type'):
f1 = sheet1.column['file']
t1 = sheet1.column['type']
s1 = sheet1.column['size, Mb']
f2 = sheet2.column['file']
t2 = sheet2.column['type']
fts1 = list(zip(f1,t1,s1))
ft2 = list(zip(f2,t2))
# Choose such elements ('file','type','size') from table 1
# whose pairs ('file','type') aren't in table 2,
# and show them.
x = [(f,t,s) for f,t,s in fts1 if (f,t) not in ft2]
print(x)
