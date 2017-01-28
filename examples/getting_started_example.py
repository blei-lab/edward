#!/usr/bin/env python
"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')


def build_toy_dataset(N=50, noise_std=0.1):
  x = np.linspace(-3, 3, num=N)
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = x.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  return x, y


def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.matmul(h, W_1) + b_1
  return tf.reshape(h, [-1])


ed.set_seed(42)

N = 50  # number of data ponts
D = 1   # number of features

# DATA
x_train, y_train = build_toy_dataset(N)

# MODEL
W_0 = Normal(mu=tf.zeros([D, 2]), sigma=tf.ones([D, 2]))
W_1 = Normal(mu=tf.zeros([2, 1]), sigma=tf.ones([2, 1]))
b_0 = Normal(mu=tf.zeros(2), sigma=tf.ones(2))
b_1 = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

x = x_train
y = Normal(mu=neural_network(x, W_0, W_1, b_0, b_1),
           sigma=0.1 * tf.ones(N))

# INFERENCE
qW_0 = Normal(mu=tf.Variable(tf.random_normal([D, 2])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, 2]))))
qW_1 = Normal(mu=tf.Variable(tf.random_normal([2, 1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([2, 1]))))
qb_0 = Normal(mu=tf.Variable(tf.random_normal([2])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([2]))))
qb_1 = Normal(mu=tf.Variable(tf.random_normal([1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data={y: y_train})


# Sample functions from variational model to visualize fits.
rs = np.random.RandomState(0)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(tf.constant(inputs), 1)
mus = []
for s in range(10):
  mus += [neural_network(x, qW_0.sample(), qW_1.sample(),
                         qb_0.sample(), qb_1.sample())]

mus = tf.stack(mus)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


# FIRST VISUALIZATION (prior)

outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0 - (CLOSE WINDOW TO CONTINUE)")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()


# RUN VARIATIONAL INFERENCE
inference.run(n_iter=500, n_samples=5)


# SECOND VISUALIZATION (posterior)

outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 1000 - (CLOSE WINDOW TO TERMINATE)")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()
