#!/usr/bin/env python
"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm
from edward.util import rbf


class BayesianNN:
  """
  Bayesian neural network for regressing outputs y on inputs x.

  p((x,y), z) = Normal(y | NN(x; z), lik_std) *
                Normal(z | 0, prior_std),

  where z are neural network weights, and with known likelihood and
  prior standard deviations.

  Parameters
  ----------
  layer_sizes : list
    The size of each layer, ordered from input to output.
  nonlinearity : function, optional
    Non-linearity after each linear transformation in the neural
    network; aka activation function.
  lik_std : float, optional
    Standard deviation of the normal likelihood; aka noise parameter,
    homoscedastic noise, scale parameter.
  prior_std : float, optional
    Standard deviation of the normal prior on weights; aka L2
    regularization parameter, ridge penalty, scale parameter.
  """
  def __init__(self, layer_sizes, nonlinearity=tf.tanh,
               lik_std=0.1, prior_std=1.0):
    self.layer_sizes = layer_sizes
    self.nonlinearity = nonlinearity
    self.lik_std = lik_std
    self.prior_std = prior_std

    self.n_layers = len(layer_sizes) - 1
    self.weight_dims = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    self.n_vars = sum((m + 1) * n for m, n in self.weight_dims)

  def neural_network(self, x, zs):
    """Forward pass of the neural net, outputting a vector of
    `n_minibatch` elements."""
    h = x
    for l in range(self.n_layers):
      W, b = zs['w' + str(l)], zs['b' + str(l)]
      h = self.nonlinearity(tf.matmul(h, W) + b)

    return tf.squeeze(h)  # n_minibatch x 1 to n_minibatch

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    x, y = xs['x'], xs['y']
    log_prior = 0.0
    for z in six.itervalues(zs):
      log_prior += tf.reduce_sum(norm.logpdf(z, 0.0, self.prior_std))

    mu = self.neural_network(x, zs)
    log_lik = tf.reduce_sum(norm.logpdf(y, mu, self.lik_std))
    return log_lik + log_prior


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  x = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = (x - 4.0) / 4.0
  x = x.reshape((N, D))
  return x, y


ed.set_seed(42)
x_train, y_train = build_toy_dataset()

model = BayesianNN(layer_sizes=[1, 10, 10, 1], nonlinearity=rbf)

qw = []
qb = []
for l in range(model.n_layers):
  m, n = model.weight_dims[l]
  qw_mu = tf.Variable(tf.random_normal([m, n]))
  qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([m, n])))
  qb_mu = tf.Variable(tf.random_normal([n]))
  qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([n])))

  qw += [Normal(mu=qw_mu, sigma=qw_sigma)]
  qb += [Normal(mu=qb_mu, sigma=qb_sigma)]

data = {'x': x_train, 'y': y_train}
inference = ed.KLqp({'w0': qw[0], 'b0': qb[0],
                     'w1': qw[1], 'b1': qb[1],
                     'w2': qw[2], 'b2': qb[2]}, data, model)
inference.run()
