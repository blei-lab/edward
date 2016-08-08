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
    X = (X - 4.0) / 4.0
    X = X.reshape((N, 1))
    return X, y.astype(np.float32)


N = 40 # num data points
p = 1 # num features

X = tf.placeholder(tf.float32, [N, p])
z = Normal([tf.zeros(p), tf.ones(p)])
y = Normal([z, tf.ones(N)],
           lambda cond_set: tf.matmul(X, cond_set[0]))

mu = tf.Variable(tf.random_normal([p]))
sigma = tf.nn.softplus(tf.Variable(tf.random_normal([p])))
qz = Normal([mu, sigma])

data = {}
data[X], data[y] = build_toy_dataset(N)

inference = ed.MFVI({z: qz}, data)
inference.initialize()

sess = ed.get_session()
for t in range(1000):
    _, loss = sess.run([inference.train, inference.loss], {X: data[X]})
    inference.print_progress(t, loss)
