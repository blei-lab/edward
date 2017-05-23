#!/usr/bin/env python
"""Wasserstein generative adversarial network for toy Gaussian data
(Arjovsky et al., 2017). A gradient penalty is used to approximate the
1-Lipschitz functional family in the Wasserstein distance (Gulrajani
et al., 2017).

Inspired by a blog post by Eric Jang.

Note there are several common failure modes, such as
(1) saturation of either discriminative or generative network;
(2) mode collapse around the true Gaussian, where the variance is
severely underestimated.

References
----------
http://edwardlib.org/tutorials/gan
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from scipy.stats import norm
from tensorflow.contrib import slim


def next_batch(N):
  samples = np.random.normal(4.0, 0.5, N)
  samples.sort()
  return samples


def generative_network(eps):
  h0 = slim.fully_connected(eps, H, activation_fn=tf.nn.relu)
  h1 = slim.fully_connected(h0, 1, activation_fn=None)
  return h1


def discriminative_network(x):
  """Outputs probability in logits."""
  h0 = slim.fully_connected(x, H * 2, activation_fn=tf.tanh)
  h1 = slim.fully_connected(h0, H * 2, activation_fn=tf.tanh)
  h2 = slim.fully_connected(h1, H * 2, activation_fn=tf.tanh)
  h3 = slim.fully_connected(h2, 1, activation_fn=None)
  return h3


def get_samples(num_points=10000, num_bins=100):
  """Return a tuple (db, pd, pg), where
  + db is the discriminator's decision boundary;
  + pd is a histogram of samples from the data distribution;
  + pg is a histogram of samples from the generative model.
  """
  sess = ed.get_session()
  bins = np.linspace(-8, 8, num_bins)

  # Decision boundary
  with tf.variable_scope("Disc", reuse=True):
    p_true = tf.sigmoid(discriminative_network(x_ph))

  xs = np.linspace(-8, 8, num_points)
  db = np.zeros((num_points, 1))
  for i in range(num_points // M):
    db[M * i:M * (i + 1)] = sess.run(
        p_true, {x_ph: np.reshape(xs[M * i:M * (i + 1)], (M, 1))})

  # Data samples
  d = next_batch(num_points)
  pd, _ = np.histogram(d, bins=bins, density=True)

  # Generated samples
  eps_ph = tf.placeholder(tf.float32, [M, 1])
  with tf.variable_scope("Gen", reuse=True):
    G = generative_network(eps_ph)

  epss = np.linspace(-8, 8, num_points)
  g = np.zeros((num_points, 1))
  for i in range(num_points // M):
    g[M * i:M * (i + 1)] = sess.run(
        G, {eps_ph: np.reshape(epss[M * i:M * (i + 1)], (M, 1))})
  pg, _ = np.histogram(g, bins=bins, density=True)

  return db, pd, pg


sns.set(color_codes=True)
ed.set_seed(42)

M = 12  # batch size during training
H = 4  # number of hidden units

# DATA. We use a placeholder to represent a minibatch. During
# inference, we generate data on the fly and feed ``x_ph``.
x_ph = tf.placeholder(tf.float32, [M, 1])

# MODEL
with tf.variable_scope("Gen"):
  eps = tf.linspace(-8.0, 8.0, M) + 0.01 * tf.random_normal([M])
  eps = tf.reshape(eps, [M, 1])
  x = generative_network(eps)

# INFERENCE
optimizer = tf.train.GradientDescentOptimizer(0.03)
optimizer_d = tf.train.GradientDescentOptimizer(0.03)

inference = ed.WGANInference(
    data={x: x_ph}, discriminator=discriminative_network)
inference.initialize(
    optimizer=optimizer, optimizer_d=optimizer_d, penalty=0.1,
    n_iter=1000)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  x_data = next_batch(M).reshape([M, 1])
  for _ in range(5):
    info_dict_d = inference.update(feed_dict={x_ph: x_data}, variables="Disc")

  info_dict = inference.update(feed_dict={x_ph: x_data}, variables="Gen")
  info_dict['t'] = info_dict['t'] // 6  # say set of 6 updates is 1 iteration
  info_dict['loss_d'] = info_dict_d['loss_d']  # get disc loss from disc update
  inference.print_progress(info_dict)

# CRITICISM
db, pd, pg = get_samples()
db_x = np.linspace(-8, 8, len(db))
p_x = np.linspace(-8, 8, len(pd))
f, ax = plt.subplots(1)
ax.plot(db_x, db, label="Decision boundary")
ax.set_ylim(0, 1)
plt.plot(p_x, pd, label="Real data")
plt.plot(p_x, pg, label="Generated data")
plt.title("1D Generative Adversarial Network")
plt.xlabel("Data values")
plt.ylabel("Probability density")
plt.legend()
plt.show()
