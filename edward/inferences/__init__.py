"""
There are two approaches to inference.

1. Idiomatic TensorFlow
  1. Build train_op (*).
  2. Build summary file writer.
  3. Build and run TensorFlow variable initializer ops.
  4. Within a training loop:
    + sess.run with infeeding and summary writers.
    + Log progress by writing to files and/or printing.
    + Check convergence (*).
  5. Build and run post-training ops (*).
2. Idiomatic TensorFlow Estimator
  + Build `model_fn` by writing a probabilistic program and calling an
  inference algorithm to produce train ops. Use the Estimator API
  workflow of `train`, `evaluate`, and `predict` alongside an
  `input_fn` data pipeline.

Inference provides utilities for both approaches. In the first
approach, it provides (*), namely: (1) inference algorithms to help
produce the train_op (and low-level functions to build your own
algorithms; sometimes post-training ops); and (2) convergence
diagnostics. In the second approach, these functions build up a
`model_fn` to form a TensorFlow Estimator.

Inference uses (unbinded) pure functions with TensorFlow idiomatic
exceptions (e.g., mutable state via TensorFlow variables; side effect
of adding to global collections and TF graph). It forgoes OO.

Specific inference files provide functions to help produce the train
(and post-training) ops.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.inferences.bigan_inference import *
from edward.inferences.conjugacy import *
from edward.inferences.gan_inference import *
from edward.inferences.hmc import *
from edward.inferences.klpq import *
from edward.inferences.klqp import *
from edward.inferences.klqp_implicit import *
from edward.inferences.laplace import *
from edward.inferences.map import *
from edward.inferences.metropolis_hastings import *
from edward.inferences.sgld import *
from edward.inferences.sghmc import *
from edward.inferences.wake_sleep import *
from edward.inferences.wgan_inference import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
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
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
