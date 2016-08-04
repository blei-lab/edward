#!/usr/bin/env python
"""
Mixture model using maximum a posteriori.

Probability model
    Mixture of Gaussians
    pi ~ Dirichlet(alpha)
    for k = 1, ..., K
        mu_k ~ N(0, cI)
        sigma_k ~ Inv-Gamma(a, b)
    for n = 1, ..., N
        c_n ~ Multinomial(pi)
        x_n|c_n ~ N(mu_{c_n}, sigma_{c_n})
Inference: Maximum a posteriori

Data: x = {x_1, ..., x_N}, where each x_i is in R^2
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.stats import dirichlet, invgamma, multivariate_normal, norm
from edward.util import get_dims, log_sum_exp


class MixtureGaussian:
    """
    Mixture of Gaussians

    p(x, z) = [ prod_{n=1}^N N(x_n; mu_{c_n}, sigma_{c_n}) Multinomial(c_n; pi) ]
              [ prod_{k=1}^K N(mu_k; 0, cI) Inv-Gamma(sigma_k; a, b) ]
              Dirichlet(pi; alpha)

    where z = {pi, mu, sigma} and for known hyperparameters a, b, c, alpha.

    Parameters
    ----------
    K : int
        Number of mixture components.
    D : float, optional
        Dimension of the Gaussians.
    """
    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.n_vars = (2*D + 1) * K

        self.a = 1
        self.b = 1
        self.c = 10
        self.alpha = tf.ones([K])

    def unpack_params(self, zs):
        """Unpack sets of parameters from a flattened matrix."""
        pi = zs[:, 0:self.K]
        mus = zs[:, self.K:(self.K+self.K*self.D)]
        sigmas = zs[:, (self.K+self.K*self.D):(self.K+2*self.K*self.D)]
        # Do the unconstrained to constrained transformation for MAP here.
        pi = tf.sigmoid(pi)
        pi = tf.concat(1, [pi[:, 0:(self.K-1)],
             tf.expand_dims(1.0 - tf.reduce_sum(pi[:, 0:(self.K-1)], 1), 0)])
        sigmas = tf.nn.softplus(sigmas)
        return pi, mus, sigmas

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        x = xs['x']
        pi, mus, sigmas = self.unpack_params(zs)
        log_prior = dirichlet.logpdf(pi, self.alpha)
        log_prior += tf.reduce_sum(norm.logpdf(mus, 0, np.sqrt(self.c)))
        log_prior += tf.reduce_sum(invgamma.logpdf(sigmas, self.a, self.b))

        # Loop over each sample zs[s, :].
        log_lik = []
        N = get_dims(x)[0]
        n_samples = get_dims(pi)[0]
        for s in range(n_samples):
            # log-likelihood is
            # sum_{n=1}^N log sum_{k=1}^K exp( log pi_k + log N(x_n; mu_k, sigma_k) )
            # Create a K x N matrix, whose entry (k, n) is
            # log pi_k + log N(x_n; mu_k, sigma_k).
            matrix = []
            for k in range(self.K):
                matrix += [tf.ones(N)*tf.log(pi[s, k]) +
                           multivariate_normal.logpdf(x,
                               mus[s, (k*self.D):((k+1)*self.D)],
                               sigmas[s, (k*self.D):((k+1)*self.D)])]

            matrix = tf.pack(matrix)
            # log_sum_exp() along the rows is a vector, whose nth
            # element is the log-likelihood of data point x_n.
            vector = log_sum_exp(matrix, 0)
            # Sum over data points to get the full log-likelihood.
            log_lik_z = tf.reduce_sum(vector)
            log_lik += [log_lik_z]

        return log_prior + tf.pack(log_lik)


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    stds = [[0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return {'x': x}


ed.set_seed(42)
data = build_toy_dataset(500)

model = MixtureGaussian(K=2, D=2)
inference = ed.MAP(model, data)
inference.run(n_iter=500, n_minibatch=10, n_print=50)
