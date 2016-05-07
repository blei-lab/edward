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
import edward as ed
import tensorflow as tf
import numpy as np

from edward.stats import dirichlet, invgamma, multivariate_normal, norm
from edward.variationals import Variational, Dirichlet, Normal, InvGamma
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
        self.num_vars = (2*D + 1) * K

        self.a = 1
        self.b = 1
        self.c = 10
        self.alpha = tf.ones([K])

    def unpack_params(self, z):
        """Unpack parameters from a flattened vector."""
        pi = z[0:self.K]
        mus = z[self.K:(self.K+self.K*self.D)]
        sigmas = z[(self.K+self.K*self.D):(self.K+2*self.K*self.D)]
        return pi, mus, sigmas

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        N = get_dims(xs)[0]
        # Loop over each mini-batch zs[b,:]
        log_prob = []
        for z in tf.unpack(zs):
            pi, mus, sigmas = self.unpack_params(z)
            log_prior = dirichlet.logpdf(pi, self.alpha)
            for k in xrange(self.K):
                log_prior += norm.logpdf(mus[k*self.D], 0, np.sqrt(self.c))
                log_prior += norm.logpdf(mus[k*self.D+1], 0, np.sqrt(self.c))
                log_prior += invgamma.logpdf(sigmas[k*self.D], self.a, self.b)
                log_prior += invgamma.logpdf(sigmas[k*self.D+1], self.a, self.b)

            log_lik = tf.constant(0.0, dtype=tf.float32)
            for x in tf.unpack(xs):
                for k in xrange(self.K):
                    log_lik += tf.log(pi[k])
                    log_lik += multivariate_normal.logpdf(x,
                        mus[(k*self.D):((k+1)*self.D)],
                        sigmas[(k*self.D):((k+1)*self.D)])

            log_prob += [log_prior + log_lik]

        return tf.pack(log_prob)

ed.set_seed(42)
x = np.loadtxt('data/mixture_data.txt', dtype='float32', delimiter=',')
data = ed.Data(tf.constant(x, dtype=tf.float32))

model = MixtureGaussian(K=2, D=2)
variational = Variational()
variational.add(Dirichlet(1, model.K))
variational.add(Normal(model.K*model.D))
variational.add(InvGamma(model.K*model.D))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=10000, n_minibatch=5, n_data=5)
