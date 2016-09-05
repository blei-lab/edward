#!/usr/bin/env python
"""
Bayesian linear regression using mean-field variational inference.

Probability model:
  Bayesian linear model
  Prior: Normal
  Likelihood: Normal
Variational model
  Likelihood: Mean-field Normal
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm


class LinearModel:
  """
  Bayesian linear regression for outputs y on inputs x.

  p((x,y), z) = Normal(y | x*z, lik_std) *
                Normal(z | 0, prior_std),

  where z are weights, and with known lik_std and prior_std.

  Parameters
  ----------
  lik_std : float, optional
    Standard deviation of the normal likelihood; aka noise parameter,
    homoscedastic noise, scale parameter.
  prior_std : float, optional
    Standard deviation of the normal prior on weights; aka L2
    regularization parameter, ridge penalty, scale parameter.
  """
  def __init__(self, lik_std=0.1, prior_std=0.1):
    self.lik_std = lik_std
    self.prior_std = prior_std

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    x, y = xs['x'], xs['y']
    w, b = zs['w'], zs['b']
    log_prior = tf.reduce_sum(norm.logpdf(w, 0.0, self.prior_std))
    log_prior += tf.reduce_sum(norm.logpdf(b, 0.0, self.prior_std))
    log_lik = tf.reduce_sum(norm.logpdf(y, ed.dot(x, w) + b, self.lik_std))
    return log_lik + log_prior

  def predict(self, xs, zs):
    """Return a prediction for each data point, via the likelihood's
    mean."""
    x = xs['x']
    w, b = zs['w'], zs['b']
    return ed.dot(x, w) + b


def build_toy_dataset(N, coeff=np.random.randn(10), noise_std=0.1):
  n_dim = len(coeff)
  x = np.random.randn(N, n_dim).astype(np.float32)
  y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=N)
  return x, y


ed.set_seed(42)

N = 40  # num data points
D = 10  # num features

coeff = np.random.randn(D)
x_train, y_train = build_toy_dataset(N, coeff)
x_test, y_test = build_toy_dataset(N, coeff)

model = LinearModel()

qw_mu = tf.Variable(tf.random_normal([D]))
qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
qb_mu = tf.Variable(tf.random_normal([]))
qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))

qw = Normal(mu=qw_mu, sigma=qw_sigma)
qb = Normal(mu=qb_mu, sigma=qb_sigma)

data = {'x': x_train, 'y': y_train}
inference = ed.MFVI({'w': qw, 'b': qb}, data, model)
inference.run(n_iter=500, n_samples=5, n_print=50)

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={'x': x_test, 'y': y_test},
                  latent_vars={'w': qw, 'b': qb}, model_wrapper=model))
