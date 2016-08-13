#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

# TODO should users do this, or use ed.*?
from edward.models import Categorical, Dirichlet, InverseGamma, Normal
from scipy.stats import norm


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    stds = [[0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


N = 500 # num data points
K = 2 # num components

pi = Dirichlet([np.array([0.1]*K)])
mu = Normal([np.zeros(K), np.zeros(K)+0.1]) # cov = 0.1 * I
sigma = InverseGamma([np.ones(K), scale=np.ones(K)])
# TODO logit form
c = Categorical([pi], lambda cond_set: tf.pack([cond_set[0] for n in range(N)]))
x = Normal([mu, sigma, c],
           lambda cond_set: cond_set[0][cond_set[2]],
           lambda cond_set: cond_set[1][cond_set[2]])


qpi_alpha = ed.softplus(tf.Variable(tf.random_normal([K])))
qmu_mu = tf.Variable(tf.random_normal([K]))
qmu_sigma = ed.softplus(tf.Variable(tf.random_normal([K])))
qsigma_alpha = ed.softplus(tf.Variable(tf.random_normal([K])))
qsigma_beta = ed.softplus(tf.Variable(tf.random_normal([K])))
qc_pi = ed.to_simplex(tf.Variable(tf.random_normal([N, K-1])))

qpi = Dirichlet([qpi_alpha])
qmu = Normal([qmu_mu, qmu_sigma])
qsigma = InverseGamma([qsigma_alpha, qsigma_beta])
# TODO logit form
qc = Categorical([qc_pi])

data = {}
data[X], data[y] = build_toy_dataset(N)

# TODO score function gradient
inference = ed.MFVI({z: qz}, data)
inference.initialize()

sess = ed.get_session()
for t in range(1000):
    _, loss = sess.run([inference.train, inference.loss], {X: data[X]})
    inference.print_progress(t, loss)
