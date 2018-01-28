"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.models.core import *
from edward.models.random_variable import *
from edward.models.queries import *

from tensorflow.python.util.all_util import remove_undocumented
from edward.models import core as _module

_allowed_symbols = [
    'RandomVariable',
    'Trace',
    'get_ancestors',
    'get_blanket',
    'get_children',
    'get_descendants',
    'get_parents',
    'get_siblings',
    'get_variables',
    'is_independent',
    'random_variables',
]
for name in dir(_module):
  obj = getattr(_module, name)
  if (isinstance(obj, type) and
          issubclass(obj, RandomVariable) and
          obj != RandomVariable):
    _allowed_symbols.append(name)

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
