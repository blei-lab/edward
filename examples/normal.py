#!/usr/bin/env python
"""
Probability model
  Posterior: (1-dimensional) Normal
Variational model
  Likelihood: Mean-field Normal
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm


class NormalPosterior:
  """p(x, z) = p(z) = p(z | x) = Normal(z; mu, sigma)"""
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma

  def log_prob(self, xs, zs):
    return norm.logpdf(zs['z'], self.mu, self.sigma)


ed.set_seed(42)
mu = tf.constant(1.0)
sigma = tf.constant(1.0)
model = NormalPosterior(mu, sigma)

qz_mu = tf.Variable(tf.random_normal([1]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

inference = ed.MFVI({'z': qz}, model_wrapper=model)
inference.run(n_iter=10000)
