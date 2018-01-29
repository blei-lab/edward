"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
This example prettifies some of the tensor naming for visualization in
TensorBoard. To view TensorBoard, run `tensorboard --logdir=log`.

References
----------
http://edwardlib.org/tutorials/bayesian-neural-network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

tf.flags.DEFINE_integer("N", default=40, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=1, help="Number of features.")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(X) + np.random.normal(0, noise_std, size=N)
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y


def main(_):
  def neural_network(X):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])
  ed.set_seed(42)

  # DATA
  X_train, y_train = build_toy_dataset(FLAGS.N)

  # MODEL
  with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([FLAGS.D, 10]), scale=tf.ones([FLAGS.D, 10]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_1")
    W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name="W_2")
    b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
    b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(FLAGS.N), name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [FLAGS.D, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, 10]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [10, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10, 10]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_2"):
      loc = tf.get_variable("loc", [10, 1])
      scale = tf.nn.softplus(tf.get_variable("scale", [10, 1]))
      qW_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_2"):
      loc = tf.get_variable("loc", [1])
      scale = tf.nn.softplus(tf.get_variable("scale", [1]))
      qb_2 = Normal(loc=loc, scale=scale)

  inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1,
                       W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
  inference.run(logdir='log')

if __name__ == "__main__":
  tf.app.run()
