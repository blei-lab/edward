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

  print("True principal axes:")
  print(w)
  return x_train


def next_batch(M):
  idx_batch = np.random.choice(N, M)
  return x_train[:, idx_batch], idx_batch


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

qz_variables = [tf.Variable(tf.random_normal([N, K])),
                tf.Variable(tf.random_normal([N, K]))]
idx_ph = ed.placeholder(tf.int32, M)
qz = Normal(
    mu=tf.transpose(tf.gather(qz_variables[0], idx_ph)),
    sigma=tf.nn.softplus(tf.transpose(tf.gather(qz_variables[1], idx_ph))))

x_ph = tf.placeholder(tf.float32, [D, M])
inference_w = ed.KLqp({w: qw}, data={x: x_ph, z: qz})
inference_z = ed.KLqp({z: qz}, data={x: x_ph, w: qw})

inference_w.initialize(scale={x: float(N) / M, z: float(N) / M},
                       var_list=qz_variables,
                       n_samples=5)
inference_z.initialize(scale={x: float(N) / M, z: float(N) / M},
                       var_list=qw_variables,
                       n_samples=5)

init = tf.initialize_all_variables()
init.run()

for _ in range(500):
  x_batch, idx_batch = next_batch(M)
  info_dict = inference_w.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})
  inference_z.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})
  inference_w.print_progress(info_dict)

sess = ed.get_session()
print("Inferred principal axes:")
print(sess.run(qw.mean()))
