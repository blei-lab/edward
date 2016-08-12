#!/usr/bin/env python
"""
This script builds the Python file `operations.py`. It generates code
using inputs from `operations.csv`.
"""
import numpy as np

def class_generator(name, op):
    op_str = \
"class NAME(DelayedOperation):\n\
    def __init__(self, *args, **kwargs):\n\
        super(NAME, self).__init__(OP, *args, **kwargs)"
    op_str = op_str.replace('NAME', name)
    op_str = op_str.replace('OP', op)
    return op_str


file_str = \
"from __future__ import absolute_import\n\
from __future__ import division\n\
from __future__ import print_function\n\
\n\
from edward.models.random_variables import DelayedOperation\n\
\n\
import six\n\
import tensorflow as tf"

operations = np.loadtxt("operations.csv", dtype=str, delimiter=',')
for name, op in operations:
    file_str += "\n\n\n"
    file_str += class_generator(name, op)

text_file = open("operations.py", "w")
text_file.write(file_str)
text_file.close()
