#!/usr/bin/env python
"""Mixture model using the Laplace approximation.

We posit a collapsed mixture of Gaussians.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import PointMass
from edward.stats import dirichlet, invgamma, multivariate_normal_diag, norm
from edward.util import get_dims, log_sum_exp


class MixtureGaussian:
  """
  Mixture of Gaussians

  p(x, z) = [ prod_{n=1}^N sum_{k=1}^K pi_k N(x_n; mu_k, sigma_k) ]
            [ prod_{k=1}^K N(mu_k; 0, cI) Inv-Gamma(sigma_k; a, b) ]
            Dirichlet(pi; alpha)

  where z = {pi, mu, sigma} and for known hyperparameters a, b, c, alpha.

  Parameters
  ----------
  K : int
    Number of mixture components.
  D : float, optional
    Dimension of the Gaussians.
  """
  def __init__(self, K, D):
    self.K = K
    self.D = D
    self.n_vars = (2 * D + 1) * K

    self.a = 1.0
    self.b = 1.0
    self.c = 3.0
    self.alpha = tf.ones([K])

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    x = xs['x']
    pi, mus, sigmas = zs['pi'], zs['mu'], zs['sigma']
    log_prior = dirichlet.logpdf(pi, self.alpha)
    log_prior += tf.reduce_sum(norm.logpdf(mus, 0.0, self.c))
    log_prior += tf.reduce_sum(invgamma.logpdf(sigmas, self.a, self.b))

    # log-likelihood is
    # sum_{n=1}^N log sum_{k=1}^K exp( log pi_k + log N(x_n; mu_k, sigma_k) )
    # Create a K x N matrix, whose entry (k, n) is
    # log pi_k + log N(x_n; mu_k, sigma_k).
    N = get_dims(x)[0]
    matrix = []
    for k in range(self.K):
      matrix += [tf.ones(N) * tf.log(pi[k]) +
                 multivariate_normal_diag.logpdf(x,
                 mus[(k * self.D):((k + 1) * self.D)],
                 sigmas[(k * self.D):((k + 1) * self.D)])]

    matrix = tf.stack(matrix)
    # log_sum_exp() along the rows is a vector, whose nth
    # element is the log-likelihood of data point x_n.
    vector = log_sum_exp(matrix, 0)
    # Sum over data points to get the full log-likelihood.
    log_lik = tf.reduce_sum(vector)

    return log_prior + log_lik


def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


ed.set_seed(42)
x_train = build_toy_dataset(500)

K = 2
D = 2
model = MixtureGaussian(K, D)

with tf.variable_scope("posterior"):
  qpi = PointMass(params=ed.to_simplex(tf.Variable(tf.random_normal([K - 1]))))
  qmu = PointMass(params=tf.Variable(tf.random_normal([K * D])))
  qsigma = PointMass(params=tf.exp(tf.Variable(tf.random_normal([K * D]))))

data = {'x': x_train}
inference = ed.Laplace({'pi': qpi, 'mu': qmu, 'sigma': qsigma}, data, model)
inference.run(n_iter=500, n_minibatch=10)
