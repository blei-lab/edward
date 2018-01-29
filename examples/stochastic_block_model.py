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


def main(_):
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
  qgamma = PointMass(tf.nn.softmax(tf.get_variable("qgamma/params", [K])))
  qPi = PointMass(tf.nn.sigmoid(tf.get_variable("qPi/params", [K, K])))
  qZ = PointMass(tf.nn.softmax(tf.get_variable("qZ/params", [N, K])))

  inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: X_data})
  inference.initialize(n_iter=250)

  tf.global_variables_initializer().run()

  for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

  # CRITICISM
  Z_pred = qZ.mean().eval().argmax(axis=1)
  print("Result (label flip can happen):")
  print("Predicted")
  print(Z_pred)
  print("True")
  print(Z_true)
  print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))

if __name__ == "__main__":
  tf.app.run()
