#!/usr/bin/env python
"""Latent space model for network data (Hoff et al., 2002).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Poisson


ed.set_seed(42)

# DATA
x_train = np.load('data/celegans_brain.npy')

# MODEL
N = x_train.shape[0]  # number of data points
K = 3  # latent dimensionality

z = Normal(mu=tf.zeros([N, K]), sigma=tf.ones([N, K]))

# Calculate N x N distance matrix.
# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2], and tile
# it to create N identical rows.
xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, N])
# 2. Create a N x N matrix where entry (i, j) is ||z_i||^2 + ||z_j||^2
# - 2 z_i^T z_j.
xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
# 3. Invert the pairwise distances and make rate along diagonals to
# be close to zero.
xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))

# Note Edward doesn't currently support sampling for Poisson.
# Hard-code it to 0's for now; it isn't used during inference.
x = Poisson(lam=xp, value=tf.zeros_like(xp))

# INFERENCE
inference = ed.MAP([z], data={x: x_train})

# Alternatively, run
# qz = Normal(mu=tf.Variable(tf.random_normal([N * K])),
#             sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N * K]))))
# inference = ed.KLqp({z: qz}, data={x: x_train})

inference.run(n_iter=2500)
