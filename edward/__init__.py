from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward import inferences
from edward import models

# Direct imports for convenience
from edward.inferences import (
    bigan_inference,
    complete_conditional,
    gan_inference,
    hmc,
    klpq,
    klqp,
    klqp_implicit,
    klqp_reparameterization,
    klqp_reparameterization_kl,
    klqp_score,
    laplace,
    map,
    metropolis_hastings,
    sghmc,
    sgld,
    wake_sleep,
    wgan_inference)
from edward.models import (
    Trace,
    get_ancestors,
    get_blanket,
    get_children,
    get_descendants,
    get_parents,
    get_siblings,
    get_variables,
    is_independent,
    random_variables)
from edward.version import __version__, VERSION

from tensorflow.python.util.all_util import remove_undocumented

# Export modules and constants.
_allowed_symbols = [
    'inferences',
    'models',
    'bigan_inference',
    'complete_conditional',
    'gan_inference',
    'hmc',
    'klpq',
    'klqp',
    'klqp_implicit',
    'klqp_reparameterization',
    'klqp_reparameterization_kl',
    'klqp_score',
    'laplace',
    'map',
    'metropolis_hastings',
    'sghmc',
    'sgld',
    'wake_sleep',
    'wgan_inference',
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
    '__version__',
    'VERSION',
]

# Remove all extra symbols that don't have a docstring or are not explicitly
# referenced in the whitelist.
remove_undocumented(__name__, _allowed_symbols, [
    inferences, models
])
