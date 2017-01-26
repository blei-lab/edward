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


def kernel(x, sigma=1.0, l=1.0):
  N = x.get_shape()[0]
  mat = []
  for i in range(N):
    vect = []
    xi = x[i, :]
    for j in range(N):
      if j == i:
        vect.append(multivariate_rbf(xi, xi, sigma, l))
      else:
        xj = x[j, :]
        vect.append(multivariate_rbf(xi, xj, sigma, l))

    mat.append(vect)

  mat = tf.pack(mat) + \
      tf.convert_to_tensor(1e-6 * np.eye(N), dtype=tf.float32)

  return mat


ed.set_seed(54)

# DATA
df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')
df[df[:, 0] == -1, 0] = 0
N = len(df)
D = df.shape[1] - 1
permutation = np.random.choice(range(N), N, replace=False)
X_train = df[:, 1:][permutation]
y_train = df[:, 0][permutation]

# MODEL
X = ed.placeholder(tf.float32, [N, D])
f = MultivariateNormalFull(mu=tf.zeros(N), sigma=kernel(X))
y = Bernoulli(logits=f)

# INFERENCE
qf = Normal(mu=tf.Variable(tf.random_normal([N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=500)
