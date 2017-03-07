#!/usr/bin/env python
"""Hierarchical logistic regression using variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal, Bernoulli
from edward.stats import bernoulli, norm


def build_toy_dataset(N, noise_std=0.1):
    D = 1
    x = np.linspace(-6, 6, num=N)
    y = np.tanh(x) + np.random.normal(0, noise_std, size=N)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    x = (x - 4.0) / 4.0
    x = x.reshape((N, D))
    return x, y


ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

x_train, y_train = build_toy_dataset(N)

x = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=3.0 * tf.ones(D))
b = Normal(mu=tf.zeros([]), sigma=3.0 * tf.ones([]))
y = Bernoulli(logits=ed.dot(x, w) + b)

qw_mu = tf.Variable(tf.random_normal([D]))
qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
qb_mu = tf.Variable(tf.random_normal([]) + 10)
qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))

qw = Normal(mu=qw_mu, sigma=qw_sigma)
qb = Normal(mu=qb_mu, sigma=qb_sigma)

sess = ed.get_session()
data = {x: x_train, y: y_train}
inference = ed.KLqp({w: qw, b: qb}, data)
inference.initialize(n_print=10, n_iter=600)

init = tf.global_variables_initializer()
init.run()

# Set up figure
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

# draws from approximate posterior
S = 50
rs = np.random.RandomState(0)
inputs = np.linspace(-5, 3, num=400, dtype=np.float32)
x_in = tf.expand_dims(inputs, 1)
mus = []
for s in range(S):
    mus += [tf.sigmoid(ed.dot(x_in, qw.sample()) + qb.sample())]
mus = tf.stack(mus)

for t in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    if t % inference.n_print == 0:
        outputs = mus.eval()

        # Plot data and functions
        plt.cla()
        ax.plot(x_train[:], y_train, 'bx')
        for s in range(S):
            ax.plot(inputs, outputs[s], alpha=0.2)
        ax.set_xlim([-5, 3])
        ax.set_ylim([-0.5, 1.5])
        plt.draw()
        plt.pause(1.0 / 60.0)
