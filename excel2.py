#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import pyexcel

# Import data from files "Untitled1.xlsx"
# and "Untitled2.xlsx":
sheet1 = pyexcel.get_sheet(file_name="Untitled5.xlsx")
sheet2 = pyexcel.get_sheet(file_name="Untitled4.xlsx")
sheet1.name_columns_by_row(0)
sheet2.name_columns_by_row(0)
# Get columns 'file', 'type' and 'size, Mb'
# from table 1 and combine them to list of triples
# ('file', 'type', 'size'). Get columns
#'file' and 'type' from
# table 2 and then combine them to list of pairs
# ('file','type'):
f1 = sheet1.column['file']
t1 = sheet1.column['type']
s1 = sheet1.column['size, Mb']
f2 = sheet2.column['file']
t2 = sheet2.column['type']
fts1 = list(zip(f1,t1,s1))
ft2 = list(zip(f2,t2))
# Choose such elements ('file','type','size')
# from table 1
# whose pairs ('file','type') aren't in table 2,
# and show them.
print("Files from table 1 which aren't in table 2:")
x = [(f,t,s) for f,t,s in fts1 if (f,t) not in ft2]
print(x)
