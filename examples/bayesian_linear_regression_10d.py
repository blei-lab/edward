#!/usr/bin/env python
"""Bayesian linear regression using mean-field variational inference.

This version uses 10 features per data point.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from scipy.stats import norm


def build_toy_dataset(N, coeff=np.random.randn(10), noise_std=0.1):
  n_dim = len(coeff)
  x = np.random.randn(N, n_dim).astype(np.float32)
  y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=N)
  return x, y


ed.set_seed(42)

N = 40  # number of data points
D = 10  # number of features

# DATA
coeff = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, coeff)
X_test, y_test = build_toy_dataset(N, coeff)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

# INFERENCE
qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

data = {X: X_train, y: y_train}
inference = ed.KLqp({w: qw, b: qb}, data)
inference.run(n_samples=5, n_iter=250)

# CRITICISM
y_post = ed.copy(y, {w: qw.mean(), b: qb.mean()})
# This is equivalent to
# y_post = Normal(mu=ed.dot(X, qw.mean()) + qb.mean(), sigma=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))
