# -*- coding: utf-8 -*-
"""
Template_script.py

What does the script do?

@author: You
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
import glob
import numpy as np
import template_library as tmp

# Reload modules
import importlib
importlib.reload(tmp)

# First thing to do in script
a = tmp.function_name_A(10)
print(a)

#FIN
