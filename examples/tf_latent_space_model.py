#!/usr/bin/env python
"""Latent space model for network data (Hoff et al., 2002).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import lognorm, norm, poisson


class LatentSpaceModel:
  """
  p(x, z) = [ prod_{i=1}^N prod_{j=1}^N Poi(Y_{ij}; 1/||z_i - z_j|| ) ]
            [ prod_{i=1}^N N(z_i; 0, I)) ]
  """
  def __init__(self, N, K, prior_std=1.0,
               like='Poisson',
               prior='Lognormal',
               dist='euclidean'):
    self.n_vars = N * K
    self.N = N
    self.K = K
    self.prior_std = prior_std
    self.like = like
    self.prior = prior
    self.dist = dist

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    if self.prior == 'Lognormal':
      log_prior = tf.reduce_sum(lognorm.logpdf(zs['z'], self.prior_std))
    elif self.prior == 'Gaussian':
      log_prior = tf.reduce_sum(norm.logpdf(zs['z'], 0.0, self.prior_std))
    else:
      raise NotImplementedError("prior not available.")

    z = tf.reshape(zs['z'], [self.N, self.K])
    if self.dist == 'euclidean':
      xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, self.N])
      xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
      xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(self.N) + 1e3))
    elif self.dist == 'cosine':
      xp = tf.matmul(z, z, transpose_b=True)

    if self.like == 'Gaussian':
      log_lik = tf.reduce_sum(norm.logpdf(xs['x'], xp, 1.0))
    elif self.like == 'Poisson':
      if not (self.dist == 'euclidean' or self.prior == "Lognormal"):
        raise NotImplementedError("Rate of Poisson has to be nonnegatve.")

      log_lik = tf.reduce_sum(poisson.logpmf(xs['x'], xp))
    else:
      raise NotImplementedError("likelihood not available.")

    return log_lik + log_prior


ed.set_seed(42)
x_train = np.load('data/celegans_brain.npy')

model = LatentSpaceModel(N=x_train.shape[0], K=3,
                         like='Poisson', prior='Gaussian')

data = {'x': x_train}
inference = ed.MAP(['z'], data, model)
# Alternatively, run
# qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
# qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
# qz = Normal(mu=qz_mu, sigma=qz_sigma)
# inference = ed.KLqp({'z': qz}, data, model)

inference.run(n_iter=2500)
