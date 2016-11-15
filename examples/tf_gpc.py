#!/usr/bin/env python
"""Gaussian process classification using mean-field variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import bernoulli, multivariate_normal_cholesky
from edward.util import multivariate_rbf_kernel

class GaussianProcess:
  """
  Gaussian process classification with sigmoid link.

  p((x,y), z) = Bernoulli(y | sigmoid(x*z)) *
                Normal(z | 0, K),

  where z are weights drawn from a GP with covariance given by 
  k(x, x') for each pair of inputs (x, x'), and with squared-exponential
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
  def __init__(self, N):
    self.N = N
    self.n_vars = N
    self.inverse_link = tf.sigmoid

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    x, y = xs['x'], xs['y']

    log_prior = multivariate_normal_cholesky.logpdf(
        zs['z'], tf.zeros(self.N), chol=tf.cholesky(K))

    log_lik = tf.reduce_sum(
        bernoulli.logpmf(y, p=self.inverse_link(y * zs['z'])))

    return log_prior + log_lik

ed.set_seed(54)

df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')
N = len(df)
permutation = np.random.choice(range(N), N, replace = False)
x = df[:, 1:][permutation]
y = df[:, 0][permutation]

print("computing the kernel matrix...")
K = multivariate_rbf_kernel(
      tf.convert_to_tensor(x), sigma=1.0, l=1.0)

data = {'x': x, 'y': y}

model = GaussianProcess(N=N)

qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

print("doing inference...")
inference = ed.MFVI({'z': qz}, data, model)
inference.run(n_iter=500)