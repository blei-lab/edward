#!/usr/bin/env python
import numpy as np
import os
from os.path import dirname
from os.path import join

def load_mixture_data():
    module_path = dirname(__file__)
    x = np.loadtxt(join(module_path, 'data', 'mixture_data.txt'), dtype='float32', delimiter=',')
    return {'x': x}
