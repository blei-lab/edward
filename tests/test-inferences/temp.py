#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Bernoulli, Normal


def four_layer_nn(x, W_1, W_2, W_3, b_1, b_2):
  h = tf.tanh(tf.matmul(x, W_1) + b_1)
  h = tf.tanh(tf.matmul(h, W_2) + b_2)
  h = tf.matmul(h, W_3)
  return tf.reshape(h, [-1])


ed.set_seed(42)

# DATA
X_train = np.zeros([500, 100])
y_train = np.zeros(500)

N = X_train.shape[0]  # data points
D = X_train.shape[1] # feature
T = 1 # number of MCMC samples

# MODEL
W_1 = Normal(mu=tf.zeros([D, 20]), sigma=tf.ones([D, 20]) * 100)
W_2 = Normal(mu=tf.zeros([20, 15]), sigma=tf.ones([20, 15]) * 100)
W_3 = Normal(mu=tf.zeros([15, 1]), sigma=tf.ones([15, 1]) * 100)
b_1 = Normal(mu=tf.zeros(20), sigma=tf.ones(20) * 100)
b_2 = Normal(mu=tf.zeros(15), sigma=tf.ones(15) * 100)

x_ph  = tf.placeholder(tf.float32,[N,D])
y = Bernoulli(logits=four_layer_nn(x_ph, W_1, W_2, W_3, b_1, b_2))

# INFERENCE
qW_1 = Empirical(params=tf.Variable(tf.random_normal([T, D, 20])))
qW_2 = Empirical(params=tf.Variable(tf.random_normal([T, 20, 15])))
qW_3 = Empirical(params=tf.Variable(tf.random_normal([T, 15, 1])))
qb_1 = Empirical(params=tf.Variable(tf.random_normal([T, 20])))
qb_2 = Empirical(params=tf.Variable(tf.random_normal([T, 15])))

inference = ed.HMC({W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2, W_3: qW_3},
                    data={y: y_train, x_ph: X_train})
inference.run(step_size=0.1)
inference = ed.SGLD({W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2, W_3: qW_3},
                    data={y: y_train, x_ph: X_train})
inference.run(step_size=0.1)
