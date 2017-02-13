#!/usr/bin/env python
"""Bayesian linear regression using variational inference.

This version directly regresses on the data X, rather than regressing
on a placeholder X. Note this prevents the model from conditioning on
other values of X.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


def build_toy_dataset(N, noise_std=0.1):
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = 5.0 * X + np.random.normal(0, noise_std, size=N)
  X = X.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  return X, y


ed.set_seed(42)

N = 40  # num data points
D = 1  # num features

# DATA
X_data, y_data = build_toy_dataset(N)

# MODEL
X = X_data
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

# INFERENCE
qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, b: qb}, data={y: y_data})
inference.run()
