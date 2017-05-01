#!/usr/bin/env python
"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.

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


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(X) + np.random.normal(0, noise_std, size=N)
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y


def neural_network(X):
  h = tf.tanh(tf.matmul(X, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return tf.reshape(h, [-1])


ed.set_seed(42)

N = 40  # number of data points
D = 1   # number of features

# DATA
X_train, y_train = build_toy_dataset(N)

# MODEL
W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]))
W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]))
W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]))
b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10))
b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10))
b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

X = tf.placeholder(tf.float32, [N, D])
y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(N))

# INFERENCE
qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 10])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 10]))))
qW_1 = Normal(loc=tf.Variable(tf.random_normal([10, 10])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([10, 10]))))
qW_2 = Normal(loc=tf.Variable(tf.random_normal([10, 1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([10, 1]))))
qb_0 = Normal(loc=tf.Variable(tf.random_normal([10])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
qb_1 = Normal(loc=tf.Variable(tf.random_normal([10])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
qb_2 = Normal(loc=tf.Variable(tf.random_normal([1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
inference.run()
