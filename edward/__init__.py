from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward import criticisms
from edward import inferences
from edward import models
# Don't explicitly import it to prevent printing deprecation warning;
# warning is displayed only if user explicitly imports it.
# from edward import stats
from edward import util

# Direct imports for convenience
from edward.criticisms import evaluate, ppc
from edward.inferences import Inference, MonteCarlo, VariationalInference, \
    HMC, MetropolisHastings, SGLD, SGHMC, \
    KLpq, KLqp, MFVI, ReparameterizationKLqp, ReparameterizationKLKLqp, \
    ReparameterizationEntropyKLqp, ScoreKLqp, ScoreKLKLqp, ScoreEntropyKLqp, \
    GANInference, WGANInference, MAP, Laplace
from edward.models import PyMC3Model, PythonModel, StanModel, \
    RandomVariable
from edward.util import copy, dot, get_ancestors, get_children, \
    get_descendants, get_dims, get_parents, get_session, get_siblings, \
    get_variables, hessian, kl_multivariate_normal, log_sum_exp, logit, \
    multivariate_rbf, placeholder, random_variables, rbf, set_seed, \
    tile, to_simplex
from edward.version import __version__
