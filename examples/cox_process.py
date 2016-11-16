#!/usr/bin/env python
"""A Cox process model for spatial analysis
(Cox, 1955; Miller et al., 2014).

The data set is a N x V matrix, where there are N data points
X={(x_1, ..., x_N)}, each x_n with a set of V counts.

We model a latent intensity function for each data point. Let K be the
N x V x V covariance matrix applied to the data set X with fixed
kernel hyperparameters, where a slice K_n is the V x V covariance
matrix over counts for a data point x_n.

For each n=1,...,N,
  p(f_n) = N(f_n | 0, K_n),
  p(x_n | f_n) = \prod_{v=1}^V p(x_{n,v} | f_{n,v}),
    where p(x_{n,v} | f_{n, v}) = Poisson(x_{n,v} | exp(f_{n,v})).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import MultivariateNormalCholesky, Normal, Poisson
from edward.util import multivariate_rbf
from scipy.stats import multivariate_normal, poisson


def kernel_row(x):
  """Compute the covariance matrix (with RBF kernel) for a data point
  x of size V, returning a matrix of shape (V, V).
  """
  sigma, l = 1.0, 1.0
  mat = []
  for i in range(V):
    vec = []
    xi = x[i]
    for j in range(V):
      if j == i:
        vec.append(multivariate_rbf(xi, xi, sigma, l))
      else:
        xj = x[j]
        vec.append(multivariate_rbf(xi, xj, sigma, l))

    mat.append(vec)

  mat = tf.pack(mat)
  # Add epsilon to ensure positive definiteness.
  return mat + tf.diag([1e-6, 1e-6])


def kernel(x):
  """It takes a data set of shape (N, V) as input and applies the
  kernel over the full data, outputting a shape of (N, V, V).
  """
  return tf.pack([kernel_row(x[n, :]) for n in range(N)])


def build_toy_dataset(N, V):
  """Generate toy data, with identity kernel for the GP."""
  K = np.identity(V)
  x = np.zeros([N, V])
  for i in range(N):
    f_i = multivariate_normal.rvs(cov=K, size=1)
    for v in range(V):
      x[i, v] = poisson.rvs(mu=np.exp(f_i[v]), size=1)

  print("Toy data:")
  print(x)
  return x

ed.set_seed(42)

N = 10  # number of data points
V = 2  # number of set counts for each data point

# DATA
x_data = build_toy_dataset(N, V)

# MODEL
x_ph = tf.placeholder(tf.float32, [N, V])

K = kernel(x_ph)  # (N, V, V) covariance, one matrix per data point
f = MultivariateNormalCholesky(mu=tf.zeros([N, V]), chol=tf.cholesky(K))

# Note Edward doesn't currently support sampling for Poisson.
# Hard-code it to 0's for now; it isn't used during inference.
x = Poisson(lam=tf.exp(f), value=tf.zeros_like(f))

# INFERENCE
qf = Normal(mu=tf.Variable(tf.random_normal([N, V])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N, V]))))

inference = ed.KLqp({f: qf}, data={x: x_data, x_ph: x_data})
inference.run(n_iter=5000)
