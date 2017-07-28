#!/usr/bin/env python
"""Stochastic Block Model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from sklearn.metrics.cluster import adjusted_rand_score
from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass

ed.set_seed(42)


def build_dataset(label_filepath, graph_filepath):
  Z = np.loadtxt(label_filepath, dtype=np.int)
  N = Z.shape[0]

  X = np.zeros((N, N))
  for line in open(graph_filepath, 'r'):
    src, dst = map(int, line.strip().split(' '))
    X[src, dst] = 1

  return X, Z


# DATA
label_filepath = 'data/karate_labels.txt'
graph_filepath = 'data/karate_edgelist.txt'
X_data, Z_true = build_dataset(label_filepath, graph_filepath)
N = X_data.shape[0]  # number of vertices
K = 2  # number of clusters

# MODEL
gamma = Dirichlet(concentration=tf.ones([K]))
Pi = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K]))
Z = Multinomial(total_count=1., probs=gamma, sample_shape=N)
X = Bernoulli(probs=tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z))))

# INFERENCE (EM algorithm)
qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([K]))))
qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, K]))))
qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, K]))))

inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: X_data})

n_iter = 100
inference.initialize(n_iter=n_iter)

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
inference.finalize()

# CRITICISM
Z_pred = qZ.mean().eval().argmax(axis=1)
print("Result (label filp can happen):")
print("Predicted")
print(Z_pred)
print("True")
print(Z_true)
print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))
