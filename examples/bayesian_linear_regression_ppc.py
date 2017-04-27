#!/usr/bin/env python
"""Bayesian linear regression using variational inference.

This version visualizes additional fits of the model.

References
----------
http://edwardlib.org/tutorials/supervised-regression
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal


def build_toy_dataset(N, noise_std=0.5):
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = 2.0 * X + 10 * np.random.normal(0, noise_std, size=N)
  X = X.reshape((N, 1))
  return X, y


ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

# DATA
X_train, y_train = build_toy_dataset(N)
X_test, y_test = build_toy_dataset(N)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

# INFERENCE
qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run()

# CRITICISM
y_post = ed.copy(y, {w: qw, b: qb})
# This is equivalent to
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Displaying prior predictive samples.")
n_prior_samples = 10

w_prior = w.sample(n_prior_samples).eval()
b_prior = b.sample(n_prior_samples).eval()

plt.scatter(X_train, y_train)

inputs = np.linspace(-1, 10, num=400)
for ns in range(n_prior_samples):
    output = inputs * w_prior[ns] + b_prior[ns]
    plt.plot(inputs, output)

plt.show()

print("Displaying posterior predictive samples.")
n_posterior_samples = 10

w_post = qw.sample(n_posterior_samples).eval()
b_post = qb.sample(n_posterior_samples).eval()

plt.scatter(X_train, y_train)

inputs = np.linspace(-1, 10, num=400)
for ns in range(n_posterior_samples):
    output = inputs * w_post[ns] + b_post[ns]
    plt.plot(inputs, output)

plt.show()
