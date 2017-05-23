#!/usr/bin/env python
"""Bayesian linear regression. Inference uses data subsampling and
scales the log-likelihood.

One local optima is an inferred posterior mean of about [-5.0 5.0].
This implies there is some weird symmetry happening; this result can
be obtained by initializing the first coordinate to be negative.
Similar occurs for the second coordinate.

Note as with all GAN-style training, the algorithm is not stable. It
is recommended to monitor training and halt manually according to some
criterion (e.g., prediction accuracy on validation test, quality of
samples).

References
----------
http://edwardlib.org/tutorials/supervised-regression
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from tensorflow.contrib import slim


def build_toy_dataset(N, w, noise_std=0.1):
  D = len(w)
  x = np.random.randn(N, D)
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  return x, y


def ratio_estimator(data, local_vars, global_vars):
  """Takes as input a dict of data x, local variable samples z, and
  global variable samples beta; outputs real values of shape
  (x.shape[0] + z.shape[0],). In this example, there are no local
  variables.
  """
  # data[y] has shape (M,); global_vars[w] has shape (D,)
  # we concatenate w to each data point y, so input has shape (M, 1 + D)
  input = tf.concat([
      tf.reshape(data[y], [M, 1]),
      tf.tile(tf.reshape(global_vars[w], [1, D]), [M, 1])], 1)
  hidden = slim.fully_connected(input, 64, activation_fn=tf.nn.relu)
  output = slim.fully_connected(hidden, 1, activation_fn=None)
  return output


def next_batch(size, i):
  diff = (i + 1) * size - X_train.shape[0]
  if diff <= 0:
    X_batch = X_train[(i * size):((i + 1) * size), :]
    y_batch = y_train[(i * size):((i + 1) * size)]
    i = i + 1
  else:
    X_batch = np.concatenate((X_train[(i * size):, :], X_train[:diff, :]))
    y_batch = np.concatenate((y_train[(i * size):], y_train[:diff]))
    i = 0

  return X_batch, y_batch, i


ed.set_seed(42)

N = 500  # number of data points
M = 50  # batch size during training
D = 2  # number of features

# DATA
w_true = np.ones(D) * 5.0
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)

# MODEL
X = tf.placeholder(tf.float32, [M, D])
y_ph = tf.placeholder(tf.float32, [M])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
y = Normal(loc=ed.dot(X, w), scale=tf.ones(M))

# INFERENCE
qw = Normal(loc=tf.Variable(tf.random_normal([D]) + 1.0),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

inference = ed.ImplicitKLqp(
    {w: qw}, data={y: y_ph},
    discriminator=ratio_estimator, global_vars={w: qw})
inference.initialize(n_iter=5000, n_print=100, scale={y: float(N) / M})

sess = ed.get_session()
tf.global_variables_initializer().run()

i = 0
for _ in range(inference.n_iter):
  X_batch, y_batch, i = next_batch(M, i)
  for _ in range(5):
    info_dict_d = inference.update(
        variables="Disc", feed_dict={X: X_batch, y_ph: y_batch})

  info_dict = inference.update(
      variables="Gen", feed_dict={X: X_batch, y_ph: y_batch})
  info_dict['loss_d'] = info_dict_d['loss_d']
  info_dict['t'] = info_dict['t'] // 6  # say set of 6 updates is 1 iteration

  t = info_dict['t']
  inference.print_progress(info_dict)
  if t == 1 or t % inference.n_print == 0:
    # Check inferred posterior parameters.
    mean, std = sess.run([qw.mean(), qw.stddev()])
    print("\nInferred mean & std:")
    print(mean)
    print(std)
