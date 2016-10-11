#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.

The model is written in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Beta
from edward.stats import bernoulli, beta


class BetaBernoulli:
  """p(x, p) = Bernoulli(x | p) * Beta(p | 1, 1)"""
  def __init__(self):
    self.n_vars = 1

  def log_prob(self, xs, zs):
    log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
    log_lik = tf.reduce_sum(bernoulli.logpmf(xs['x'], p=zs['p']))
    return log_lik + log_prior


ed.set_seed(42)
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

model = BetaBernoulli()

inference = ed.MAP(['p'], data, model)
inference.run(n_iter=100)
