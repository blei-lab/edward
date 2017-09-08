#!/usr/bin/env python
"""Stochastic block model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass
from observations import karate
from sklearn.metrics.cluster import adjusted_rand_score

ed.set_seed(42)

# DATA
X_data, Z_true = karate("~/data")
N = X_data.shape[0]  # number of vertices
K = 2  # number of clusters

# MODEL
gamma = Dirichlet(concentration=tf.ones([K]))
Pi = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K]))
Z = Multinomial(total_count=1.0, probs=gamma, sample_shape=N)
X = Bernoulli(probs=tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z))))

# INFERENCE (EM algorithm)
qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([K]))))
qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, K]))))
qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, K]))))

inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: X_data})

n_iter = 250
inference.initialize(n_iter=n_iter)

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

inference.finalize()

# CRITICISM
Z_pred = qZ.mean().eval().argmax(axis=1)
print("Result (label flip can happen):")
print("Predicted")
print(Z_pred)
print("True")
print(Z_true)
print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))
