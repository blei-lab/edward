"""Assessments for program and inference correctness.

We can never validate whether a model is true. In practice, ``all
models are wrong'' [@box1976science]. However, we can try to
uncover where the model goes wrong. Model criticism helps justify the
model as an approximation or point to good directions for revising the
model. For background, see the criticism [tutorial](/tutorials/criticism).

Edward explores model criticism using

+ point evaluations, such as mean squared error or
  classification accuracy;
+ posterior predictive checks, for making probabilistic
  assessments of the model fit using discrepancy functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.criticisms.evaluate import *
from edward.criticisms.ppc import *
from edward.criticisms.ppc_plots import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'evaluate',
    'ppc',
    'ppc_density_plot',
    'ppc_stat_hist_plot',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
