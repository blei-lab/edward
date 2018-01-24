"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.samplers.rejection import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'GammaRejectionSampler',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)