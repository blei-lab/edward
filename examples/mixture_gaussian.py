#!/usr/bin/env python
"""
Mixture model using mean-field variational inference.

Probability model:
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

from edward.variationals import Likelihood, MFDirichlet, MFGaussian, MFInvGamma
class MFMixGaussian(Likelihood):
    """
    q(z | lambda ) = q(pi) prod_{k=1}^K q(mu_k) q(sigma_k)
        q(pi) = Dirichlet(alpha')
        q(mu_k) = N(mu'_k, Sigma'_k)
        q(sigma_k) = Inv-Gamma(a'_k, b'_k)

    where z = {pi, mu, sigma}
    """
    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.dirichlet = MFDirichlet(1, K)
        self.gaussian = MFGaussian(K*D)
        self.invgamma = MFInvGamma(K*D)

        Likelihood.__init__(self, self.dirichlet.num_vars + \
                                  self.gaussian.num_vars +  \
                                  self.invgamma.num_vars)
        self.num_params = self.dirichlet.num_params + \
                          self.gaussian.num_params + \
                          self.invgamma.num_params

    def mapping(self, x):
        return [self.dirichlet.mapping(x),
                self.gaussian.mapping(x),
                self.invgamma.mapping(x)]

    def set_params(self, params):
    	self.dirichlet.set_params(params[0])
        self.gaussian.set_params(params[1])
        self.invgamma.set_params(params[2])

    def print_params(self, sess):
    	self.dirichlet.print_params(sess)
        self.gaussian.print_params(sess)
        self.invgamma.print_params(sess)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        z_dirichlet = self.dirichlet.sample((size[0], self.dirichlet.num_vars), sess)
        z_gaussian = sess.run(self.gaussian.sample((size[0], self.gaussian.num_vars), sess))
        z_invgamma = self.invgamma.sample((size[0], self.invgamma.num_vars), sess)
        return np.concatenate((z_dirichlet, z_gaussian, z_invgamma), axis=1)

    def log_prob_zi(self, i, z):
        """log q(z_i | lambda_i)"""
        if i < self.dirichlet.num_vars:
            return self.dirichlet.log_prob_zi(i, z[:, 0:self.dirichlet.num_vars])
        elif i < self.dirichlet.num_vars + self.gaussian.num_vars:
            i = i - self.dirichlet.num_vars
            return self.gaussian.log_prob_zi(i,
                       z[:, self.dirichlet.num_vars:(self.dirichlet.num_vars+self.gaussian.num_vars)])
        elif i < self.num_vars:
            i = i - self.dirichlet.num_vars - self.gaussian.num_vars
            return self.invgamma.log_prob_zi(i,
                       z[:, (self.dirichlet.num_vars+self.gaussian.num_vars):])
        else:
            raise

ed.set_seed(42)
x = np.loadtxt('data/mixture_data.txt', dtype='float32', delimiter=',')
data = ed.Data(tf.constant(x, dtype=tf.float32))

model = MixtureGaussian(K=2, D=2)
variational = MFMixGaussian(model.K, model.D)
inference = ed.MFVI(model, variational, data)
inference.run(n_iter=10000, n_minibatch=5)
