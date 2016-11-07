#!/usr/bin/env python
"""Probabilistic matrix factorization.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Normal, PointMass
from edward.stats import lognorm, norm, poisson


class MatrixFactorization:
  """
  p(x, z) = [ prod_{i=1}^N prod_{j=1}^N Poi(Y_{ij}; \exp(s_iTt_j) ) ]
            [ prod_{i=1}^N N(s_i; 0, var) N(t_i; 0, var) ]

  where z = {s,t}.
  """
  def __init__(self, K, n_rows, n_cols=None, prior_std=0.1,
               like='Poisson',
               prior='Lognormal',
               interaction='additive'):
    if n_cols is None:
       n_cols = n_rows
    self.n_vars = (n_rows + n_cols) * K
    self.n_rows = n_rows
    self.n_cols = n_cols
    self.K = K
    self.prior_std = prior_std
    self.like = like
    self.prior = prior
    self.interaction = interaction

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    if self.prior == 'Lognormal':
      log_prior = tf.reduce_sum(lognorm.logpdf(zs['z'], self.prior_std))
    elif self.prior == 'Gaussian':
      log_prior = tf.reduce_sum(norm.logpdf(zs['z'], 0.0, self.prior_std))
    else:
      raise NotImplementedError("prior not available.")

    s = tf.reshape(zs['z'][:self.n_rows * self.K], [self.n_rows, self.K])
    t = tf.reshape(zs['z'][self.n_cols * self.K:], [self.n_cols, self.K])

    xp = tf.matmul(s, t, transpose_b=True)
    if self.interaction == 'multiplicative':
      xp = tf.exp(xp)
    elif self.interaction != 'additive':
      raise NotImplementedError("interaction type unknown.")

    if self.like == 'Gaussian':
      log_lik = tf.reduce_sum(norm.logpdf(xs['x'], xp, 1.0))
    elif self.like == 'Poisson':
      if not (self.interaction == "additive" or self.prior == "Lognormal"):
        raise NotImplementedError("Rate of Poisson has to be nonnegatve.")

      log_lik = tf.reduce_sum(poisson.logpmf(xs['x'], xp))
    else:
      raise NotImplementedError("likelihood not available.")

    return log_lik + log_prior


ed.set_seed(42)
x_train = np.load('data/celegans_brain.npy')

K = 3
model = MatrixFactorization(K, n_rows=x_train.shape[0])

qz = PointMass(
    params=tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars]))))

data = {'x': x_train}
inference = ed.MAP({'z': qz}, data, model)
# Alternatively, run
# qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
# qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
# qz = Normal(mu=qz_mu, sigma=qz_sigma)
# inference = ed.KLqp({'z': qz}, data, model)

inference.run(n_iter=2500)
