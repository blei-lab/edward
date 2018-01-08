from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward import criticisms
from edward import inferences
from edward import models
from edward import util

# Direct imports for convenience
from edward.criticisms import (
    evaluate, ppc, ppc_density_plot, ppc_stat_hist_plot)
from edward.inferences import (
    bigan_inference,
    complete_conditional,
    gan_inference,
    implicit_klqp,
    klpq,
    klqp,
    reparameterization_klqp,
    reparameterization_kl_klqp,
    reparameterization_entropy_klqp,
    score_klqp,
    score_kl_klqp,
    score_entropy_klqp,
    score_rb_klqp,
    laplace,
    map,
    wake_sleep,
    wgan_inference)
# from edward.inferences import MonteCarlo, HMC, MetropolisHastings, SGLD, SGHMC, Gibbs
from edward.models import RandomVariable
from edward.util import (
    copy, dot,
    get_ancestors, get_blanket, get_children, get_control_variate_coef,
    get_descendants, get_parents, get_session, get_siblings, get_variables,
    is_independent, Progbar, random_variables, rbf, set_seed,
    to_simplex, transform)
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
    'bigan_inference',
    'complete_conditional',
    'gan_inference',
    'implicit_klqp',
    'MonteCarlo',
    'HMC',
    'MetropolisHastings',
    'SGLD',
    'SGHMC',
    'klpq',
    'klqp',
    'reparameterization_klqp',
    'reparameterization_kl_klqp',
    'reparameterization_entropy_klqp',
    'score_klqp',
    'score_kl_klqp',
    'score_entropy_klqp',
    'score_rb_klqp',
    'laplace',
    'map',
    'wake_sleep',
    'wgan_inference',
    'Gibbs',
    'RandomVariable',
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
    'is_independent',
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
