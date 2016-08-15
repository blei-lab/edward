#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from scipy.stats import norm

sg = tf.contrib.bayesflow.stochastic_graph


def build_toy_dataset(N=40, noise_std=0.1):
    ed.set_seed(0)
    X  = np.concatenate([np.linspace(0, 2, num=N/2),
                         np.linspace(6, 8, num=N/2)])
    y = 5.0*X + norm.rvs(0, noise_std, size=N)
    X = X.reshape((N, 1))
    return X.astype(np.float32), y.astype(np.float32)


N = 40 # num data points
p = 1 # num features

ed.set_seed(42)

X = tf.placeholder(tf.float32, [N, p])
tf.add_to_collection('placeholders', X)
with sg.value_type(sg.SampleValue(n=1)):
    beta = Normal(mu=tf.zeros(p), sigma=tf.ones(p))

# We require (input_size, n_samples) for y, so do (input_size, p) %*%
# (p, n_samples).
# TODO does this apply to non-stochastic inference approaches?
y = Normal(mu=tf.matmul(X, beta, transpose_b=True), sigma=tf.ones(p))
#y = Normal(mu=tf.matmul(X, tf.expand_dims(beta, 1)), sigma=tf.ones(p))

qmu_mu = tf.Variable(tf.random_normal([p]))
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([p])))
with sg.value_type(sg.SampleValue(n=1)):
    qbeta = Normal(mu=qmu_mu, sigma=qmu_sigma)

X_data, y_data = build_toy_dataset(N)
data = {X: X_data, y: y_data}

inference = ed.MFVI({beta: qbeta}, data)
inference.initialize()

sess = ed.get_session()
for t in range(501):
    _, loss = sess.run([inference.train, inference.loss], {X: data[X]})
    inference.print_progress(t, loss)
