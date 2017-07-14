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


def build_dataset(Z, Pi):
    return np.random.binomial(1, p=Z.dot(Pi).dot(Z.T))


# DATA
gamma_true = np.array([0.3, 0.4, 0.3])
Pi_true = np.array([[0.9, 0.1, 0.1],
                    [0.1, 0.1, 0.7],
                    [0.1, 0.7, 0.1]])

N = 50  # number of vertices
K = gamma_true.shape[0]  # number of clusters

Z_true = np.random.multinomial(1, pvals=gamma_true, size=N)

X_data = build_dataset(Z_true, Pi_true)

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

n_iter = 300
inference.initialize(n_iter=n_iter)

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
inference.finalize()

# CRITICISM
predicted = qZ.mean().eval().argmax(axis=1)
answer = Z_true.argmax(axis=1)
print("Result (label filp can happen):")
print("Predicted")
print(predicted)
print("True")
print(answer)
print("Adjusted Rand Index =", adjusted_rand_score(predicted, answer))
