#!/usr/bin/env python
"""Mixture density network using maximum likelihood.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from edward.stats import norm
from sklearn.model_selection import train_test_split


class MixtureDensityNetwork:
  def __init__(self, K):
    self.K = K

  def neural_network(self, X):
    """pi, mu, sigma = NN(x; theta)"""
    hidden1 = slim.fully_connected(X, 25)
    hidden2 = slim.fully_connected(hidden1, 25)
    self.pi = slim.fully_connected(hidden2, self.K, activation_fn=tf.nn.softmax)
    self.mus = slim.fully_connected(hidden2, self.K, activation_fn=None)
    self.sigmas = slim.fully_connected(hidden2, self.K,
                                       activation_fn=tf.nn.softplus)

  def log_prob(self, xs, zs):
    """Return scalar, the log joint density log p(xs, zs)."""
    # Note there are no parameters we're being Bayesian about. The
    # parameters are baked into how we specify the neural networks.
    X, y = xs['X'], xs['y']
    self.neural_network(X)
    result = self.pi * norm.prob(y, self.mus, self.sigmas)
    result = tf.log(tf.reduce_sum(result, 1))
    return tf.reduce_sum(result)


def build_toy_dataset(N):
  y_data = np.random.uniform(-10.5, 10.5, (N, 1)).astype(np.float32)
  r_data = np.random.normal(size=(N, 1)).astype(np.float32)  # random noise
  x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
  return train_test_split(x_data, y_data, random_state=42)


ed.set_seed(42)

N = 6000  # number of data points
D = 1  # number of features

# DATA
X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

X = tf.placeholder(tf.float32, [None, D])
y = tf.placeholder(tf.float32, [None, D])
data = {'X': X, 'y': y}

# MODEL
model = MixtureDensityNetwork(10)

# INFERENCE
inference = ed.MAP([], data, model)
sess = ed.get_session()
inference.initialize()

init = tf.global_variables_initializer()
init.run()

n_epoch = 20
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X: X_train, y: y_train})
  train_loss[i] = info_dict['loss']
  test_loss[i] = sess.run(inference.loss, feed_dict={X: X_test, y: y_test})
  print("Train Loss: {:0.3f}, Test Loss: {:0.3f}".format(
      train_loss[i], test_loss[i]))
