# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:37:20 2018

@author: lasayr
"""
from os import listdir
from os.path import isfile, join
from analysisFunctions import sjoin

ld2 = ld
ld2.append("sfsf")

for f in ld2:
    print(f)
    if isfile(sjoin(path, f)):
        print("yesfile")
    else:
        print("nosfile")
            
