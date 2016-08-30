#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

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


N = 25 # num data points
K = 2 # num components
D = 2 # dimensionality of data
ed.set_seed(42)

pi = Dirichlet(alpha=tf.constant([0.1]*K), name='pi') # (K, )
mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]), name='mu') # (K, D)
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]), name='sigma') # (K, D)
c = Categorical(logits=tf.pack([ed.logit(pi) for i in range(N)]), name='c') # (N, )
x = Normal(mu=tf.gather(mu, c), sigma=tf.gather(sigma, c), name='x') # (N, D)

qpi_alpha = tf.nn.softplus(tf.Variable(tf.random_normal([K])))
qmu_mu = tf.Variable(tf.random_normal([K, D]))
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([K, D])))
qsigma_alpha = tf.nn.softplus(tf.Variable(tf.random_normal([K, D])))
qsigma_beta = tf.nn.softplus(tf.Variable(tf.random_normal([K, D])))
qc_logits = tf.Variable(tf.random_normal([N, K])) # TODO technically only need [N, K-1]

qpi = Dirichlet(alpha=qpi_alpha, name='qpi') # (K, )
qmu = Normal(mu=qmu_mu, sigma=qmu_sigma, name='qmu') # (K, D)
qsigma = InverseGamma(alpha=qsigma_alpha, beta=qsigma_beta, name='qsigma') # (K, D)
qc = Categorical(logits=qc_logits, name='qc') # (N, )

data = {x: build_toy_dataset(N)}

# TODO numerical instability and convergence problems
inference = ed.MFVI({pi: qpi, mu: qmu, sigma: qsigma, c: qc}, data)
inference.initialize(n_samples=10, logdir='train')

sess = ed.get_session()
for t in range(5001):
  _, loss = sess.run([inference.train, inference.loss])
  if t % inference.n_print == 0:
    print("iter {:d} loss {:.2f}".format(t, loss))
    print("Inferred membership probabilities:")
    print(sess.run(qpi.mean()))
    print("Inferred cluster means:")
    print(sess.run(qmu.mean()))
    print("Inferred cluster standard deviations:")
    print(sess.run(qmu.std()))
