"""Bayesian logistic regression using Hamiltonian Monte Carlo.

We visualize the fit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Empirical

tf.flags.DEFINE_integer("N", default=40, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=1, help="Number of features.")
tf.flags.DEFINE_integer("T", default=5000, help="Number of posterior samples.")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(N, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 6, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y


def main(_):
  ed.set_seed(42)

  # DATA
  X_train, y_train = build_toy_dataset(FLAGS.N)

  # MODEL
  X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D])
  w = Normal(loc=tf.zeros(FLAGS.D), scale=3.0 * tf.ones(FLAGS.D))
  b = Normal(loc=tf.zeros([]), scale=3.0 * tf.ones([]))
  y = Bernoulli(logits=ed.dot(X, w) + b)

  # INFERENCE
  qw = Empirical(params=tf.get_variable("qw/params", [FLAGS.T, FLAGS.D]))
  qb = Empirical(params=tf.get_variable("qb/params", [FLAGS.T]))

  inference = ed.HMC({w: qw, b: qb}, data={X: X_train, y: y_train})
  inference.initialize(n_print=10, step_size=0.6)

  # Alternatively, use variational inference.
  # qw_loc = tf.get_variable("qw_loc", [FLAGS.D])
  # qw_scale = tf.nn.softplus(tf.get_variable("qw_scale", [FLAGS.D]))
  # qb_loc = tf.get_variable("qb_loc", []) + 10.0
  # qb_scale = tf.nn.softplus(tf.get_variable("qb_scale", []))

  # qw = Normal(loc=qw_loc, scale=qw_scale)
  # qb = Normal(loc=qb_loc, scale=qb_scale)

  # inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
  # inference.initialize(n_print=10, n_iter=600)

  tf.global_variables_initializer().run()

  # Set up figure.
  fig = plt.figure(figsize=(8, 8), facecolor='white')
  ax = fig.add_subplot(111, frameon=False)
  plt.ion()
  plt.show(block=False)

  # Build samples from inferred posterior.
  n_samples = 50
  inputs = np.linspace(-5, 3, num=400, dtype=np.float32).reshape((400, 1))
  probs = tf.stack([tf.sigmoid(ed.dot(inputs, qw.sample()) + qb.sample())
                    for _ in range(n_samples)])

  for t in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    if t % inference.n_print == 0:
      outputs = probs.eval()

      # Plot data and functions
      plt.cla()
      ax.plot(X_train[:], y_train, 'bx')
      for s in range(n_samples):
        ax.plot(inputs[:], outputs[s], alpha=0.2)

      ax.set_xlim([-5, 3])
      ax.set_ylim([-0.5, 1.5])
      plt.draw()
      plt.pause(1.0 / 60.0)

if __name__ == "__main__":
  tf.app.run()
