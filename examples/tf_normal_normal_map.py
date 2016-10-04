#!/usr/bin/env python
"""Normal-normal model using MAP estimation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.stats import norm


class NormalModel:
  """p(x, z) = Normal(x; z, sigma) Normal(z; mu, sigma)"""
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma
    self.n_vars = 1

  def log_prob(self, xs, zs):
    log_prior = norm.logpdf(zs['z'], self.mu, self.sigma)
    log_lik = tf.reduce_sum(norm.logpdf(xs['x'], zs['z'], self.sigma))
    return log_lik + log_prior


ed.set_seed(42)
data = {'x': np.array([3] * 20 + [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        dtype=np.float32)}

mu = tf.constant(3.0)
sigma = tf.constant(0.1)
model = NormalModel(mu, sigma)

inference = ed.MAP(['z'], data, model)
inference.run(n_iter=200)
