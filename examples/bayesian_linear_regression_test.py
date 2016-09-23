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


def build_toy_dataset(N, coeff=np.random.randn(10), noise_std=0.1):
  n_dim = len(coeff)
  x = np.random.randn(N, n_dim).astype(np.float32)
  y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=N)
  return x, y


ed.set_seed(42)

N = 40  # num data points
D = 10  # num features

# DATA
coeff = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, coeff)
X_test, y_test = build_toy_dataset(N, coeff)

# MODEL
X = ed.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

# INFERENCE
qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

data = {X: X_train, y: y_train}
inference = ed.MFVI({w: qw, b: qb}, data)
inference.initialize(n_samples=5, n_print=50)

sess = ed.get_session()
for t in range(251):
  _, loss = sess.run([inference.train, inference.loss], {X: data[X]})
  inference.print_progress(t, loss)

# CRITICISM
y_post = ed.copy(y, {w: qw.mean(), b: qb.mean()})
# This is equivalent to
# y_post = Normal(mu=ed.dot(X, qw.mean()) + qb.mean(), sigma=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))
