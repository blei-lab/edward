#!/usr/bin/env python
"""Probabilistic principal components analysis (Tipping and Bishop, 1999).

Inference uses data subsampling.
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


def next_batch(M):
  return x_train[:, np.random.choice(N, M)]


ed.set_seed(142)

N = 5000  # number of data points
M = 100  # minibatch size
D = 2  # data dimensionality
K = 1  # latent dimensionality

# DATA

x_train = build_toy_dataset(N, D, K)

# MODEL

w = Normal(mu=tf.zeros([D, K]), sigma=10.0 * tf.ones([D, K]))
z = Normal(mu=tf.zeros([K, M]), sigma=tf.ones([K, M]))
x = Normal(mu=tf.matmul(w, z), sigma=tf.ones([D, M]))

# INFERENCE

qw_variables = [tf.Variable(tf.random_normal([D, K])),
                tf.Variable(tf.random_normal([D, K]))]
qw = Normal(mu=qw_variables[0], sigma=tf.nn.softplus(qw_variables[1]))

qz_variables = [tf.Variable(tf.random_normal([K, M])),
                tf.Variable(tf.random_normal([K, M]))]
qz = Normal(mu=qz_variables[0], sigma=tf.nn.softplus(qz_variables[1]))

x_ph = tf.placeholder(tf.float32, [D, M])
inference = ed.KLqp({w: qw, z: qz}, data={x: x_ph})

with tf.variable_scope("optimizer"):
  inference.initialize(scale={x: float(N) / M, z: float(N) / M})

init = tf.initialize_all_variables()
init.run()

init_local = tf.initialize_variables(
    qz_variables + tf.get_collection(tf.GraphKeys.VARIABLES, scope="optimizer"))

for _ in range(250):
  x_batch = next_batch(M)
  for _ in range(50):
    inference.update(feed_dict={x_ph: x_batch})

  # Reinitialize local variables and also adaptive optimizer's history.
  init_local.run()

sess = ed.get_session()
print("Principal axes:")
print(sess.run(qw.mean()))
