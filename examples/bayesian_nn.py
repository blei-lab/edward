#!/usr/bin/env python
"""
Bayesian neural network using mean-field variational inference.
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016))
Inspired by autograd's Bayesian neural network example.

Probability model:
  Bayesian neural network
  Prior: Normal
  Likelihood: Normal with mean parameterized by fully connected NN
Variational model
  Likelihood: Mean-field Normal
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm
from edward.util import rbf


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  x = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(x) + norm.rvs(0, noise_std, size=N)
  x = (x - 4.0) / 4.0
  x = x.reshape((N, D))
  return x, y


def neural_network(x):
    h = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.nn.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])


ed.set_seed(42)

N = 40  # num data ponts
D = 1   # num features

# DATA
x_train, y_train = build_toy_dataset(N)

# MODEL
W_0 = Normal(mu=tf.zeros([D, 10]), sigma=tf.ones([D, 10]))
W_1 = Normal(mu=tf.zeros([10, 10]), sigma=tf.ones([10, 10]))
W_2 = Normal(mu=tf.zeros([10, 1]), sigma=tf.ones([10, 1]))
b_0 = Normal(mu=tf.zeros(10), sigma=tf.ones(10))
b_1 = Normal(mu=tf.zeros(10), sigma=tf.ones(10))
b_2 = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

x = tf.convert_to_tensor(x_train, dtype=tf.float32)
y = Normal(mu=neural_network(x), sigma=0.1 * tf.ones(N))

# INFERENCE
qW_0 = Normal(mu=tf.Variable(tf.random_normal([D, 10])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, 10]))))
qW_1 = Normal(mu=tf.Variable(tf.random_normal([10, 10])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([10, 10]))))
qW_2 = Normal(mu=tf.Variable(tf.random_normal([10, 1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([10, 1]))))
qb_0 = Normal(mu=tf.Variable(tf.random_normal([10])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
qb_1 = Normal(mu=tf.Variable(tf.random_normal([10])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
qb_2 = Normal(mu=tf.Variable(tf.random_normal([1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

data = {y: y_train}
inference = ed.MFVI({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data)
inference.initialize(n_print=100)

init = tf.initialize_all_variables()
init.run()

for _ in range(1000):
  info_dict = inference.update()
  t, loss = info_dict['t'], info_dict['loss']
  if t % inference.n_print == 0:
    print("iter {:d} loss {:.2f}".format(t, loss))
