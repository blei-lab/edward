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

tf.flags.DEFINE_integer("N", default=50, help="Number of users.")
tf.flags.DEFINE_integer("M", default=60, help="Number of movies.")
tf.flags.DEFINE_integer("D", default=3, help="Number of latent factors.")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(U, V, N, M, noise_std=0.1):
  R = np.dot(np.transpose(U), V) + np.random.normal(0, noise_std, size=(N, M))
  return R


def get_indicators(N, M, prob_std=0.5):
  ind = np.random.binomial(1, prob_std, (N, M))
  return ind


def main(_):
  # true latent factors
  U_true = np.random.randn(FLAGS.D, FLAGS.N)
  V_true = np.random.randn(FLAGS.D, FLAGS.M)

  # DATA
  R_true = build_toy_dataset(U_true, V_true, FLAGS.N, FLAGS.M)
  I_train = get_indicators(FLAGS.N, FLAGS.M)
  I_test = 1 - I_train

  # MODEL
  I = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.M])
  U = Normal(loc=0.0, scale=1.0, sample_shape=[FLAGS.D, FLAGS.N])
  V = Normal(loc=0.0, scale=1.0, sample_shape=[FLAGS.D, FLAGS.M])
  R = Normal(loc=tf.matmul(tf.transpose(U), V) * I,
             scale=tf.ones([FLAGS.N, FLAGS.M]))

  # INFERENCE
  qU = Normal(loc=tf.get_variable("qU/loc", [FLAGS.D, FLAGS.N]),
              scale=tf.nn.softplus(
                  tf.get_variable("qU/scale", [FLAGS.D, FLAGS.N])))
  qV = Normal(loc=tf.get_variable("qV/loc", [FLAGS.D, FLAGS.M]),
              scale=tf.nn.softplus(
                  tf.get_variable("qV/scale", [FLAGS.D, FLAGS.M])))

  inference = ed.KLqp({U: qU, V: qV}, data={R: R_true, I: I_train})
  inference.run()

  # CRITICISM
  qR = Normal(loc=tf.matmul(tf.transpose(qU), qV),
              scale=tf.ones([FLAGS.N, FLAGS.M]))

  print("Mean squared error on test data:")
  print(ed.evaluate('mean_squared_error', data={qR: R_true, I: I_test}))

  plt.imshow(R_true, cmap='hot')
  plt.show()

  R_est = tf.matmul(tf.transpose(qU), qV).eval()
  plt.imshow(R_est, cmap='hot')
  plt.show()

if __name__ == "__main__":
  tf.app.run()
