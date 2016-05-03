#!/usr/bin/env python
"""
TODO this doesn't work
This is the implementation of the Bayesian Mixture of K Gaussians.
The model is written in Stan.
Data: X = {x1,...,xn} where each xi is in R^d..we choose d=2 in our example
Probability model
    Likelihood:
         x_i|c_i ~ N(mu_{c_i}, sigma_{c_i})
         c_i ~ Discrete(theta)
    Prior:
         theta ~ Dirichlet(alpha_0) where alpha_0 is a vector of dimension K
         mu_j ~ N(0, cI) iid...we take c = 10 in our example
         sigma_j ~ inverse_gamma(a, b) iid..we take a = b = 1 in our example
Variational model
    q(pi) ~ Dirichlet(alpha') where alpha' is a vector of dimension K
    q(mu_j) ~ N(mj', Sigmaj') iid
    q(sigma_j) ~ inverse_gamma(aj', Bj') iid
    q(c_i) ~ Multinomial(phi_i)  iid...integrated out
"""
import tensorflow as tf
import edward as ed
import numpy as np
from edward.stats import dirichlet, norm, invgamma

class GaussMixture:
    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.num_vars = (2*D + 1) * K

    def log_prob(self, xs, zs):
        alpha_vec = np.ones(K)
        dirich_log_prior = dirichlet.logpdf(zs[:,0:K], alpha=alpha_vec)
        gauss_log_prior = norm.logpdf(zs[:, K:(K+K*D)], loc=0, scale=np.sqrt(10))
        invgam_log_prior = invgamma.logpdf(zs[:, K+K*D:(K+2*K*D)], alpha=1, beta=1)

        log_prior = dirich_log_prior + gauss_log_prior + invgam_log_prior

        log_lik = 0

        return log_lik + log_prior

ed.set_seed(42)
K = 2
D = 2
model = GaussMixture(K, D)
variational = ed.MFMixGaussian(D, K)
x = np.loadtxt('data/mix_mock_data.txt', dtype='float32', delimiter=',')
N = len(x)
data = ed.Data(tf.constant(x, dtype=tf.float32))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=1000)
