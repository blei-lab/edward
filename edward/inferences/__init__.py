"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.inferences.bigan_inference import *
from edward.inferences.conjugacy import *
from edward.inferences.gan_inference import *
from edward.inferences.gibbs import *
from edward.inferences.hmc import *
from edward.inferences.implicit_klqp import *
from edward.inferences.inference import *
from edward.inferences.klpq import *
from edward.inferences.klqp import *
from edward.inferences.laplace import *
from edward.inferences.map import *
from edward.inferences.metropolis_hastings import *
from edward.inferences.monte_carlo import *
from edward.inferences.replica_exchange_mc import *
from edward.inferences.sgld import *
from edward.inferences.sghmc import *
from edward.inferences.variational_inference import *
from edward.inferences.wake_sleep import *
from edward.inferences.wgan_inference import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'BiGANInference',
    'complete_conditional',
    'GANInference',
    'Gibbs',
    'HMC',
    'ImplicitKLqp',
    'Inference',
    'KLpq',
    'KLqp',
    'ReparameterizationKLqp',
    'ReparameterizationKLKLqp',
    'ReparameterizationEntropyKLqp',
    'ReplicaExchangeMC',
    'ScoreKLqp',
    'ScoreKLKLqp',
    'ScoreEntropyKLqp',
    'ScoreRBKLqp',
    'Laplace',
    'MAP',
    'MetropolisHastings',
    'MonteCarlo',
    'SGLD',
    'SGHMC',
    'VariationalInference',
    'WakeSleep',
    'WGANInference',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
