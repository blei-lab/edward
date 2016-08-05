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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Variational, InvGamma, Normal
from edward.stats import bernoulli, invgamma, multivariate_normal
from edward.util import multivariate_rbf
from edward.datasets import load_crabs_data


class GaussianProcess:
    """
    Gaussian process classification

    p((x,y), z) = Bernoulli(y | logit^{-1}(w*z)) *
                  Normal(w | 0, K) * InvGamma(l | 1, 1)

    where z are weights drawn from a GP with covariance given by k(x,
    x') for each pair of inputs (x, x'), and with squared-exponential
    kernel and known kernel hyperparameters.

    Parameters
    ----------
    N : int
        Number of data points.
    l : float, optional
        Length scale parameter.
    """
    def __init__(self, N, sigma=1.0):
        self.N = N
        self.sigma = sigma

        self.n_vars = N
        self.inverse_link = tf.sigmoid

    def kernel(self, x, l):
        mat = []
        for i in range(self.N):
            mat += [[]]
            xi = x[i, :]
            for j in range(self.N):
                if j == i:
                    mat[i] += [multivariate_rbf(xi, xi, self.sigma, l)]
                else:
                    xj = x[j, :]
                    mat[i] += [multivariate_rbf(xi, xj, self.sigma, l)]

            mat[i] = tf.pack(mat[i])

        return tf.pack(mat)

    def log_prob(self, xs, zs):
        """Return a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        x, y = xs['x'], xs['y']
        ws, ls = zs
        log_prior = tf.pack(
            [tf.reduce_sum(multivariate_normal.logpdf(ws, cov=self.kernel(x, tf.squeeze(l))))
             for l in tf.unpack(ls)])
        log_prior += tf.reduce_sum(invgamma.logpdf(ls, 1.0, 1.0), 1)
        log_lik = tf.pack([tf.reduce_sum(
            bernoulli.logpmf(y, self.inverse_link(tf.mul(y, w)))
            ) for w in tf.unpack(ws)])
        return log_prior + log_lik


ed.set_seed(42)

data = load_crabs_data(N=25)

model = GaussianProcess(N=len(data['x']))
variational = Variational()
variational.add(Normal(model.n_vars))
variational.add(InvGamma(1))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=500, n_samples=5)
