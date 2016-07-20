#!/usr/bin/env python
"""
Mixture model using mean-field variational inference.

Probability model
    Mixture of Gaussians
    pi ~ Dirichlet(alpha)
    for k = 1, ..., K
        mu_k ~ N(0, cI)
        sigma_k ~ Inv-Gamma(a, b)
    for n = 1, ..., N
        c_n ~ Multinomial(pi)
        x_n|c_n ~ N(mu_{c_n}, sigma_{c_n})
Variational model
    Likelihood:
        q(pi) prod_{k=1}^K q(mu_k) q(sigma_k)
        q(pi) = Dirichlet(alpha')
        q(mu_k) = N(mu'_k, Sigma'_k)
        q(sigma_k) = Inv-Gamma(a'_k, b'_k)
    (We collapse the c_n latent variables in the probability model's
    joint density.)

Data: x = {x_1, ..., x_N}, where each x_i is in R^2
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Variational, Dirichlet, Normal, InvGamma
from edward.stats import dirichlet, invgamma, multivariate_normal, norm
from edward.util import get_dims


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

    def log_prob(self, xs, zs):
        """Return a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        N = get_dims(xs['x'])[0]
        pi, mus, sigmas = zs
        log_prior = dirichlet.logpdf(pi, self.alpha)
        log_prior += tf.reduce_sum(norm.logpdf(mus, 0, np.sqrt(self.c)), 1)
        log_prior += tf.reduce_sum(invgamma.logpdf(sigmas, self.a, self.b), 1)

        # Loop over each sample zs[b,:]
        log_lik = []
        n_samples = get_dims(zs[0])[0]
        for s in range(n_samples):
            log_lik_z = N*tf.reduce_sum(tf.log(pi), 1)
            for k in range(self.K):
                log_lik_z += tf.reduce_sum(multivariate_normal.logpdf(xs['x'],
                    mus[s, (k*self.D):((k+1)*self.D)],
                    sigmas[s, (k*self.D):((k+1)*self.D)]))

            log_lik += [log_lik_z]

        return log_prior + tf.pack(log_lik)


ed.set_seed(42)
x = np.loadtxt('data/mixture_data.txt', dtype='float32', delimiter=',')
data = {'x': x}

model = MixtureGaussian(K=2, D=2)
variational = Variational()
variational.add(Dirichlet(model.K))
variational.add(Normal(model.K*model.D))
variational.add(InvGamma(model.K*model.D))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=500, n_samples=5, n_minibatch=5)
