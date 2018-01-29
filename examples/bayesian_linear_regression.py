"""Bayesian linear regression using stochastic gradient Hamiltonian
Monte Carlo.

This version visualizes additional fits of the model.

References
----------
http://edwardlib.org/tutorials/supervised-regression
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical

tf.flags.DEFINE_integer("N", default=40, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=1, help="Number of features.")
tf.flags.DEFINE_integer("T", default=5000, help="Number of posterior samples.")
tf.flags.DEFINE_integer("nburn", default=100,
                        help="Number of burn-in samples.")
tf.flags.DEFINE_integer("stride", default=10,
                        help="Frequency with which to plots samples.")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(N, noise_std=0.5):
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = 2.0 * X + 10 * np.random.normal(0, noise_std, size=N)
  X = X.reshape((N, 1))
  return X, y


def main(_):
  ed.set_seed(42)

  # DATA
  X_train, y_train = build_toy_dataset(FLAGS.N)
  X_test, y_test = build_toy_dataset(FLAGS.N)

  # MODEL
  X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D])
  w = Normal(loc=tf.zeros(FLAGS.D), scale=tf.ones(FLAGS.D))
  b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
  y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(FLAGS.N))

  # INFERENCE
  qw = Empirical(params=tf.get_variable("qw/params", [FLAGS.T, FLAGS.D]))
  qb = Empirical(params=tf.get_variable("qb/params", [FLAGS.T, 1]))

  inference = ed.SGHMC({w: qw, b: qb}, data={X: X_train, y: y_train})
  inference.run(step_size=1e-3)

  # CRITICISM

  # Plot posterior samples.
  sns.jointplot(qb.params.eval()[FLAGS.nburn:FLAGS.T:FLAGS.stride],
                qw.params.eval()[FLAGS.nburn:FLAGS.T:FLAGS.stride])
  plt.show()

  # Posterior predictive checks.
  y_post = ed.copy(y, {w: qw, b: qb})
  # This is equivalent to
  # y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(FLAGS.N))

  print("Mean squared error on test data:")
  print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

  print("Displaying prior predictive samples.")
  n_prior_samples = 10

  w_prior = w.sample(n_prior_samples).eval()
  b_prior = b.sample(n_prior_samples).eval()

  plt.scatter(X_train, y_train)

  inputs = np.linspace(-1, 10, num=400)
  for ns in range(n_prior_samples):
      output = inputs * w_prior[ns] + b_prior[ns]
      plt.plot(inputs, output)

  plt.show()

  print("Displaying posterior predictive samples.")
  n_posterior_samples = 10

  w_post = qw.sample(n_posterior_samples).eval()
  b_post = qb.sample(n_posterior_samples).eval()

  plt.scatter(X_train, y_train)

  inputs = np.linspace(-1, 10, num=400)
  for ns in range(n_posterior_samples):
      output = inputs * w_post[ns] + b_post[ns]
      plt.plot(inputs, output)

  plt.show()

if __name__ == "__main__":
  tf.app.run()
