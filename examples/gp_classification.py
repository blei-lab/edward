#!/usr/bin/env python
"""
Gaussian process classification using mean-field variational inference.

Probability model:
    Gaussian process classification
    Prior: Gaussian process
    Likelihood: Bernoulli-Logit
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational, Normal
from edward.stats import bernoulli, multivariate_normal
from edward.util import multivariate_rbf

class GaussianProcess:
    """
    Gaussian process classification

    p((x,y), z) = Bernoulli(y | logit^{-1}(x*z)) *
                  Normal(z | 0, K),

    where z are weights drawn from a GP with covariance given by k(x,
    x') for each pair of inputs (x, x'), and with squared-exponential
    kernel and known kernel hyperparameters.

    Parameters
    ----------
    N : int
        Number of data points.
    sigma : float, optional
        Signal variance parameter.
    l : float, optional
        Length scale parameter.
    """
    def __init__(self, N, sigma=1.0, l=1.0):
        self.N = N
        self.sigma = sigma
        self.l = l

        self.num_vars = N
        self.inverse_link = tf.sigmoid

    def kernel(self, x):
        mat = []
        for i in range(self.N):
            mat += [[]]
            xi = x[i, :]
            for j in range(self.N):
                if j == i:
                    mat[i] += [multivariate_rbf(xi, xi, self.sigma, self.l)]
                else:
                    xj = x[j, :]
                    mat[i] += [multivariate_rbf(xi, xj, self.sigma, self.l)]

            mat[i] = tf.pack(mat[i])

        return tf.pack(mat)

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        x, y = xs['x'], xs['y']
        log_prior = multivariate_normal.logpdf(zs, cov=self.kernel(x))
        log_lik = tf.pack([tf.reduce_sum(
            bernoulli.logpmf(y, self.inverse_link(tf.mul(y, z)))
            ) for z in tf.unpack(zs)])
        return log_prior + log_lik

ed.set_seed(42)
df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')[:25, :]
data = {'x': tf.constant(df[:, 1:], dtype=tf.float32),
        'y': tf.constant(df[:, 0], dtype=tf.float32)}

model = GaussianProcess(N=len(df))
variational = Variational()
variational.add(Normal(model.num_vars))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=500)
