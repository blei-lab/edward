#!/usr/bin/env python
"""Bayesian linear regression using variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm


class LinearModel:
  """
  Bayesian linear regression for outputs y on inputs x.

  p((x,y), (w,b)) = Normal(y | x*w + b, lik_std) *
                    Normal(w | 0, prior_std) *
                    Normal(b | 0, prior_std),

  where w and b are weights and intercepts, and with known lik_std and
  prior_std.

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
    x, y = xs['x'], xs['y']
    w, b = zs['w'], zs['b']
    log_prior = tf.reduce_sum(norm.logpdf(w, 0.0, self.prior_std))
    log_prior += tf.reduce_sum(norm.logpdf(b, 0.0, self.prior_std))
    log_lik = tf.reduce_sum(norm.logpdf(y, ed.dot(x, w) + b, self.lik_std))
    return log_lik + log_prior


def build_toy_dataset(N, noise_std=0.1):
  x = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = 0.075 * x + np.random.normal(0, noise_std, size=N)
  x = (x - 4.0) / 4.0
  x = x.reshape((N, 1))
  return x, y


ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

x_train, y_train = build_toy_dataset(N)

model = LinearModel()

qw_mu = tf.Variable(tf.random_normal([D]))
qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
qb_mu = tf.Variable(tf.random_normal([]))
qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))

qw = Normal(mu=qw_mu, sigma=qw_sigma)
qb = Normal(mu=qb_mu, sigma=qb_sigma)

# Set up figure
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

sess = ed.get_session()
data = {'x': x_train, 'y': y_train}
inference = ed.KLqp({'w': qw, 'b': qb}, data, model)
inference.initialize(n_samples=5, n_iter=250, n_print=5)

init = tf.global_variables_initializer()
init.run()

for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  if t % inference.n_print == 0:
    # Sample functions from variational model
    w_mean, w_std = sess.run([qw.mu, qb.sigma])
    b_mean, b_std = sess.run([qb.mu, qb.sigma])
    rs = np.random.RandomState(0)
    ws = (rs.randn(1, 10) * w_std + w_mean).astype(np.float32)
    bs = (rs.randn(10) * b_std + b_mean).astype(np.float32)
    inputs = np.linspace(-8, 8, num=400, dtype=np.float32)
    x = tf.expand_dims(inputs, 1)
    mus = tf.matmul(x, ws) + bs
    outputs = mus.eval()

    # Get data
    x, y = data['x'], data['y']

    # Plot data and functions
    plt.cla()
    ax.plot(x, y, 'bx')
    ax.plot(inputs, outputs)
    ax.set_ylim([-2, 3])
    plt.draw()
    plt.pause(1.0 / 60.0)
