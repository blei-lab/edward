#!/usr/bin/env python
"""Hierarchical logistic regression using variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import bernoulli, norm


class HierarchicalLogistic:
  """
  Hierarchical logistic regression for outputs y on inputs x.

  p((x,y), z) = Bernoulli(y | link^{-1}(x*z)) *
                Normal(z | 0, prior_std),

  where z are weights, and with known link function and
  prior_variance.

  Parameters
  ----------
  weight_dim : list
    Dimension of weights, which is dimension of input x dimension
    of output.
  inv_link : function, optional
    Inverse of link function, which is applied to the linear transformation.
  prior_std : float, optional
    Standard deviation of the normal prior on weights; aka L2
    regularization parameter, ridge penalty, scale parameter.
  """
  def __init__(self, inv_link=tf.sigmoid, prior_std=3.0):
    self.inv_link = inv_link
    self.prior_std = prior_std

  def log_prob(self, xs, zs):
    x, y = xs['x'], xs['y']
    w, b = zs['w'], zs['b']
    log_prior = tf.reduce_sum(norm.logpdf(w, 0.0, self.prior_std))
    log_prior += tf.reduce_sum(norm.logpdf(b, 0.0, self.prior_std))
    log_lik = tf.reduce_sum(bernoulli.logpmf(y,
                            p=self.inv_link(ed.dot(x, w) + b)))
    return log_lik + log_prior


def build_toy_dataset(N, noise_std=0.1):
  D = 1
  x = np.linspace(-3, 3, num=N)
  y = np.tanh(x) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  x = (x - 4.0) / 4.0
  x = x.reshape((N, D))
  return x, y


ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

x_train, y_train = build_toy_dataset(N)

model = HierarchicalLogistic()

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
inference.initialize(n_print=5, n_iter=600)

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
    inputs = np.linspace(-3, 3, num=400, dtype=np.float32)
    x = tf.expand_dims(inputs, 1)
    ps = model.inv_link(tf.matmul(x, ws) + bs)
    outputs = ps.eval()

    # Get data
    x, y = data['x'], data['y']

    # Plot data and functions
    plt.cla()
    ax.plot(x, y, 'bx')
    ax.plot(inputs, outputs)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-0.5, 1.5])
    plt.draw()
    plt.pause(1.0 / 60.0)
