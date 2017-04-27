#!/usr/bin/env python
"""Bayesian linear regression using variational inference.

This version directly regresses on the data X, rather than regressing
on a placeholder X. Note this prevents the model from conditioning on
other values of X.

References
----------
http://edwardlib.org/tutorials/supervised-regression
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
  X = X.reshape((N, 1))
  return X, y


ed.set_seed(42)

N = 40  # num data points
D = 1  # num features

# DATA
X_data, y_data = build_toy_dataset(N)

# MODEL
X = tf.cast(X_data, tf.float32)
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

# INFERENCE
qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, b: qb}, data={y: y_data})
inference.run()
