#!/usr/bin/env python
"""Correlated normal posterior.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal
from edward.stats import multivariate_normal
from edward.util import get_dims


class NormalPosterior:
  """p(x, z) = p(z) = p(z | x) = Normal(z; mu, sigma)"""
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma
    self.n_vars = get_dims(mu)[0]

  def log_prob(self, xs, zs):
    return multivariate_normal.logpdf(zs['z'], self.mu, self.sigma)


ed.set_seed(42)
mu = tf.constant([1.0, 1.0])
sigma = tf.constant([[1.0, 0.1],
                     [0.1, 1.0]])
model = NormalPosterior(mu, sigma)

qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

inference = ed.KLqp({'z': qz}, model_wrapper=model)
inference.run(n_iter=300)
