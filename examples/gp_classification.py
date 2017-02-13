#!/usr/bin/env python
"""Gaussian process classification using variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, MultivariateNormalFull, Normal
from edward.util import multivariate_rbf


def kernel(x):
  mat = []
  for i in range(N):
    mat += [[]]
    xi = x[i, :]
    for j in range(N):
      if j == i:
        mat[i] += [multivariate_rbf(xi, xi)]
      else:
        xj = x[j, :]
        mat[i] += [multivariate_rbf(xi, xj)]

    mat[i] = tf.stack(mat[i])

  return tf.stack(mat)


ed.set_seed(42)

# DATA
df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')
df[df[:, 0] == -1, 0] = 0  # replace -1 label with 0 label
N = 25  # number of data points
D = df.shape[1] - 1  # number of features
subset = np.random.choice(df.shape[0], N, replace=False)
X_train = df[subset, 1:]
y_train = df[subset, 0]

# MODEL
X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalFull(mu=tf.zeros(N), sigma=kernel(X))
y = Bernoulli(logits=f)

# INFERENCE
qf = Normal(mu=tf.Variable(tf.random_normal([N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=500)
