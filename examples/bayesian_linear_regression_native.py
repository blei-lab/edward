#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from scipy.stats import norm


def build_toy_dataset(N=40, noise_std=0.1):
    ed.set_seed(0)
    X  = np.concatenate([np.linspace(0, 2, num=N/2),
                         np.linspace(6, 8, num=N/2)])
    y = 5.0*X + norm.rvs(0, noise_std, size=N)
    X = X.reshape((N, 1))
    return X, y.astype(np.float32)


N = 40 # num data points
p = 1 # num features

ed.set_seed(42)

X = tf.placeholder(tf.float32, [N, p])
beta = Normal(mu=tf.zeros(p), sigma=tf.ones(p))
# We require (input_size, n_samples) for y, so do (input_size, p) %*%
# (p, n_samples).
# TODO does this apply to non-stochastic inference approaches?
y = Normal(mu=ed.matmul(X, beta, transpose_b=True), sigma=tf.ones(p))

data = {}
data[X], data[y] = build_toy_dataset(N)

mu = tf.Variable(tf.random_normal([p]))
sigma = tf.nn.softplus(tf.Variable(tf.random_normal([p])))
qbeta = Normal(mu=mu, sigma=sigma)

inference = ed.MFVI({beta: qbeta}, data)
inference.initialize()

sess = ed.get_session()
for t in range(500):
    _, loss = sess.run([inference.train, inference.loss], {X: data[X]})
    inference.print_progress(t, loss)
