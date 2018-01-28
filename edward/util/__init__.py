"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.util.random_variables import *
from edward.util.tensorflow import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'copy',
    'get_ancestors',
    'get_blanket',
    'get_children',
    'get_control_variate_coef',
    'get_descendants',
    'get_parents',
    'get_siblings',
    'get_variables',
    'is_independent',
    'transform',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
