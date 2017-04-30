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


def ratio_estimator(data, local_vars, global_vars):
  # data[y] has shape (M,); global_vars[w] has shape (D,)
  # we concatenate w to each data point y, so input has shape (M, 1 + D)
  input = tf.concat([
      tf.reshape(data[y], [M, 1]),
      tf.tile(tf.reshape(global_vars[w], [1, 1]), [M, 1])], 1)
  # this is equivalent to previous. it has the error, meaning it is
  # not slim. the initializations are the same, as iter 0 is same
  # hidden = slim.fully_connected(input, 64, activation_fn=None,
  #     weights_initializer=tf.random_normal_initializer(),
  #     biases_initializer=None)
  # output = slim.fully_connected(hidden, 1, activation_fn=None,
  #     weights_initializer=tf.random_normal_initializer(),
  #     biases_initializer=None)
  w1 = tf.get_variable("w1", shape=[2, 64],
      initializer=tf.random_normal_initializer())
  w2 = tf.get_variable("w2", shape=[64, 1],
      initializer=tf.random_normal_initializer())
  hidden = tf.matmul(input, w1)
  output = tf.matmul(hidden, w2)
  return output


ed.set_seed(42)

M = 50  # size of data

# DATA
w_true = 5.0
X_data = np.random.randn(M)
y_data = X_data * w_true + np.random.normal(0, 0.1, size=M)

# MODEL
X = tf.cast(X_data, tf.float32)
w = Normal(loc=0.0, scale=1.0)
y = Normal(loc=X * w, scale=tf.ones(M))

# INFERENCE
qw = Normal(loc=tf.Variable(1.0), scale=1.0)

inference = ed.ImplicitKLqp(
    {w: qw}, data={y: y_data},
    discriminator=ratio_estimator, global_vars={w: qw})
inference.initialize(n_iter=50, n_print=10)

sess = ed.get_session()
tf.global_variables_initializer().run()

# with tf.variable_scope("Disc", reuse=True):
#   w2 = tf.get_variable("w2")
#   print(w2.eval().reshape([64]))

for _ in range(inference.n_iter):
  info_dict = inference.update(variables="Disc")
  t = info_dict['t']
  if t == 1 or t % inference.n_print == 0:
    print(info_dict['loss_d'])
