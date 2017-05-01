#!/usr/bin/env python
"""Mixture of Gaussians.

Perform inference with Metropolis-Hastings. It utterly fails. This is
because we are proposing a sample in a high-dimensional space. The
acceptance ratio is so small that it is unlikely we'll ever accept a
proposed sample. A Gibbs-like extension ("MH within Gibbs"), which
does a separate MH in each dimension, may succeed.

References
----------
http://edwardlib.org/tutorials/unsupervised
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import \
    Categorical, Dirichlet, Empirical, InverseGamma, Normal
from scipy.stats import norm


def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2))
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


N = 500  # number of data points
K = 2  # number of components
D = 2  # dimensionality of data
ed.set_seed(42)

# DATA
x_data = build_toy_dataset(N)

# MODEL
pi = Dirichlet(concentration=tf.constant([1.0] * K))
mu = Normal(loc=tf.zeros([K, D]), scale=tf.ones([K, D]))
sigma = InverseGamma(concentration=tf.ones([K, D]), rate=tf.ones([K, D]))
c = Categorical(logits=tf.tile(tf.reshape(ed.logit(pi), [1, K]), [N, 1]))
x = Normal(loc=tf.gather(mu, c), scale=tf.gather(sigma, c))

# INFERENCE
T = 5000
qpi = Empirical(params=tf.Variable(tf.ones([T, K]) / K))
qmu = Empirical(params=tf.Variable(tf.zeros([T, K, D])))
qsigma = Empirical(params=tf.Variable(tf.ones([T, K, D])))
qc = Empirical(params=tf.Variable(tf.zeros([T, N], dtype=tf.int32)))

gpi = Dirichlet(concentration=tf.constant([1.4, 1.6]))
gmu = Normal(loc=tf.constant([[1.0, 1.0], [-1.0, -1.0]]),
             scale=tf.constant([[0.5, 0.5], [0.5, 0.5]]))
gsigma = InverseGamma(concentration=tf.constant([[1.1, 1.1], [1.1, 1.1]]),
                      rate=tf.constant([[1.0, 1.0], [1.0, 1.0]]))
gc = Categorical(logits=tf.zeros([N, K]))

inference = ed.MetropolisHastings(
    latent_vars={pi: qpi, mu: qmu, sigma: qsigma, c: qc},
    proposal_vars={pi: gpi, mu: gmu, sigma: gsigma, c: gc},
    data={x: x_data})

inference.initialize()

sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  t = info_dict['t']
  if t == 1 or t % inference.n_print == 0:
    qpi_mean, qmu_mean = sess.run([qpi.mean(), qmu.mean()])
    print("")
    print("Inferred membership probabilities:")
    print(qpi_mean)
    print("Inferred cluster means:")
    print(qmu_mean)
