"""There are four modules in Edward:

+ [`ed.criticisms`](/api/ed/criticisms)
  is comprised of functions. They operate on
  random variables in a model or they operate on NumPy arrays
  representing values drawn from the random variables.
+ [`ed.inferences`](/api/ed/inferences)
  is mostly comprised of classes. They are
  organized in a class hierarchy, where methods are shared via parent
  classes and `Inference` is the top-most base class.
+ [`ed.models`](/api/ed/models)
  is comprised of random variables.
  The list of available random variables depends on the TensorFlow
  version installed. The module's API docs describe those available for
  TensorFlow {{tensorflow_version}}.
+ [`ed.util`](/api/ed/util)
  is comprised of functions for miscellaneous usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward import criticisms
from edward import inferences
from edward import models
from edward import util

# Direct imports for convenience
from edward.criticisms import evaluate, ppc, ppc_density_plot, \
    ppc_stat_hist_plot
from edward.inferences import Inference, MonteCarlo, VariationalInference, \
    HMC, MetropolisHastings, SGLD, SGHMC, \
    KLpq, KLqp, ReparameterizationKLqp, ReparameterizationKLKLqp, \
    ReparameterizationEntropyKLqp, ScoreKLqp, ScoreKLKLqp, ScoreEntropyKLqp, \
    ScoreRBKLqp, WakeSleep, GANInference, BiGANInference, WGANInference, \
    ImplicitKLqp, MAP, Laplace, complete_conditional, Gibbs
from edward.models import RandomVariable
from edward.util import check_data, check_latent_vars, copy, dot, \
    get_ancestors, get_blanket, get_children, get_control_variate_coef, \
    get_descendants, get_parents, get_session, get_siblings, get_variables, \
    Progbar, random_variables, rbf, set_seed, to_simplex, transform
from edward.version import __version__, VERSION

from tensorflow.python.util.all_util import remove_undocumented

# Export modules and constants.
_allowed_symbols = [
    'criticisms',
    'inferences',
    'models',
    'util',
    'evaluate',
    'ppc',
    'ppc_density_plot',
    'ppc_stat_hist_plot',
    'Inference',
    'MonteCarlo',
    'VariationalInference',
    'HMC',
    'MetropolisHastings',
    'SGLD',
    'SGHMC',
    'KLpq',
    'KLqp',
    'ReparameterizationKLqp',
    'ReparameterizationKLKLqp',
    'ReparameterizationEntropyKLqp',
    'ScoreKLqp',
    'ScoreKLKLqp',
    'ScoreEntropyKLqp',
    'ScoreRBKLqp',
    'WakeSleep',
    'GANInference',
    'BiGANInference',
    'WGANInference',
    'ImplicitKLqp',
    'MAP',
    'Laplace',
    'complete_conditional',
    'Gibbs',
    'RandomVariable',
    'check_data',
    'check_latent_vars',
    'copy',
    'dot',
    'get_ancestors',
    'get_blanket',
    'get_children',
    'get_control_variate_coef',
    'get_descendants',
    'get_parents',
    'get_session',
    'get_siblings',
    'get_variables',
    'Progbar',
    'random_variables',
    'rbf',
    'set_seed',
    'to_simplex',
    'transform',
    '__version__',
    'VERSION',
]

# Remove all extra symbols that don't have a docstring or are not explicitly
# referenced in the whitelist.
remove_undocumented(__name__, _allowed_symbols, [
    criticisms, inferences, models, util
])
