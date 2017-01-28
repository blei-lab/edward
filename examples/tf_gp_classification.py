#!/usr/bin/env python
"""Gaussian process classification using variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import bernoulli, multivariate_normal
from edward.util import multivariate_rbf


class GaussianProcess:
  """
  Gaussian process classification

  p((x,y), z) = Bernoulli(y | logit^{-1}(x*z)) *
                Normal(z | 0, K),

  where z are weights drawn from a GP with covariance given by k(x,
  x') for each pair of inputs (x, x'), and with squared-exponential
  kernel and known kernel hyperparameters.

  Parameters
  ----------
  N : int
    Number of data points.
  sigma : float, optional
    Signal variance parameter.
  l : float, optional
    Length scale parameter.
  """
  def __init__(self, N, sigma=1.0, l=1.0):
    self.N = N
    self.sigma = sigma
    self.l = l

    self.n_vars = N
    self.inverse_link = tf.sigmoid

  def kernel(self, x):
    mat = []
    for i in range(self.N):
      mat += [[]]
      xi = x[i, :]
      for j in range(self.N):
        if j == i:
          mat[i] += [multivariate_rbf(xi, xi, self.sigma, self.l)]
        else:
          xj = x[j, :]
          mat[i] += [multivariate_rbf(xi, xj, self.sigma, self.l)]

      mat[i] = tf.stack(mat[i])

    return tf.stack(mat)

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    x, y = xs['x'], xs['y']
    log_prior = multivariate_normal.logpdf(
        zs['z'], tf.zeros(self.N), self.kernel(x))
    log_lik = tf.reduce_sum(
        bernoulli.logpmf(y, p=self.inverse_link(y * zs['z'])))
    return log_prior + log_lik


ed.set_seed(42)

df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')
N = 25  # number of data points
subset = np.random.choice(df.shape[0], N, replace=False)
X_train = df[subset, 1:]
y_train = df[subset, 0]

model = GaussianProcess(N)

qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

data = {'x': X_train, 'y': y_train}
inference = ed.KLqp({'z': qz}, data, model)
inference.run(n_iter=500)
