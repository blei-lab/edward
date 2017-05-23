#!/usr/bin/env python
"""Probabilistic matrix factorization using variational inference.

Visualizes the actual and the estimated rating matrices as heatmaps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal


def build_toy_dataset(U, V, N, M, noise_std=0.1):
  R = np.dot(np.transpose(U), V) + np.random.normal(0, noise_std, size=(N, M))
  return R


def get_indicators(N, M, prob_std=0.5):
  ind = np.random.binomial(1, prob_std, (N, M))
  return ind


N = 50  # number of users
M = 60  # number of movies
D = 3  # number of latent factors

# true latent factors
U_true = np.random.randn(D, N)
V_true = np.random.randn(D, M)

# DATA
R_true = build_toy_dataset(U_true, V_true, N, M)
I_train = get_indicators(N, M)
I_test = 1 - I_train

# MODEL
I = tf.placeholder(tf.float32, [N, M])
U = Normal(loc=tf.zeros([D, N]), scale=tf.ones([D, N]))
V = Normal(loc=tf.zeros([D, M]), scale=tf.ones([D, M]))
R = Normal(loc=tf.matmul(tf.transpose(U), V) * I, scale=tf.ones([N, M]))

# INFERENCE
qU = Normal(loc=tf.Variable(tf.random_normal([D, N])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, N]))))
qV = Normal(loc=tf.Variable(tf.random_normal([D, M])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, M]))))

inference = ed.KLqp({U: qU, V: qV}, data={R: R_true, I: I_train})
inference.run()

# CRITICISM
qR = Normal(loc=tf.matmul(tf.transpose(qU), qV), scale=tf.ones([N, M]))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={qR: R_true, I: I_test}))

plt.imshow(R_true, cmap='hot')
plt.show()

R_est = tf.matmul(tf.transpose(qU), qV).eval()
plt.imshow(R_est, cmap='hot')
plt.show()
