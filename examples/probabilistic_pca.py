#!/usr/bin/env python
"""Probabilistic principal components analysis (Tipping and Bishop, 1999).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from scipy.stats import norm


def build_toy_dataset(N, D, K, sigma=1):
  x_train = np.zeros((D, N))
  w = norm.rvs(loc=0, scale=2, size=(D, K))
  z = norm.rvs(loc=0, scale=1, size=(K, N))
  mean = np.dot(w, z)
  for d in range(D):
    for n in range(N):
      x_train[d, n] = norm.rvs(loc=mean[d, n], scale=sigma)

  return x_train


ed.set_seed(142)

N = 5000  # number of data points
D = 2  # data dimensionality
K = 1  # latent dimensionality

# DATA

x_train = build_toy_dataset(N, D, K)

# MODEL

w = Normal(mu=tf.zeros([D, K]), sigma=10.0 * tf.ones([D, K]))
z = Normal(mu=tf.zeros([K, N]), sigma=tf.ones([K, N]))
x = Normal(mu=tf.matmul(w, z), sigma=tf.ones([D, N]))

# INFERENCE

qw = Normal(mu=tf.Variable(tf.random_normal([D, K])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qz = Normal(mu=tf.Variable(tf.random_normal([K, N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([K, N]))))

inference = ed.KLqp({w: qw, z: qz}, data={x: x_train})

init = tf.initialize_all_variables()
inference.run(n_iter=500, n_print=100, n_samples=10)

sess = ed.get_session()
print("Principal axes:")
print(sess.run(qw.mean()))
