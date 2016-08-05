#!/usr/bin/env python
import numpy as np
import os
from os.path import dirname
from os.path import join

def load_mixture_data():
    module_path = dirname(__file__)
    x = np.loadtxt(join(module_path, 'data', 'mixture_data.txt'), dtype='float32', delimiter=',')
    return {'x': x}

def load_crabs_data(N=None):
    module_path = dirname(__file__)
    df = np.loadtxt(join(module_path, 'data', 'crabs_train.txt'), dtype='float32', delimiter=',')
    if type(N)!= int or N>len(df):
        N = len(df)
    return {'x': df[:N, 1:], 'y': df[:N, 0]}

def load_celegans_brain():
    module_path = dirname(__file__)
    x = np.load(join(module_path, 'data', 'celegans_brain.npy'))
    return {'x': x}

