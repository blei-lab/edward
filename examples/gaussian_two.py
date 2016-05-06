#!/usr/bin/env python
"""
Probability model
    Posterior: (2-dimensional) Gaussian
Variational model
    Likelihood: Mean-field Gaussian
"""
import edward as ed
import tensorflow as tf

from edward.stats import multivariate_normal
from edward.util import get_dims

class Gaussian:
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, Sigma)
    """
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma
        self.num_vars = get_dims(mu)[0]

    def log_prob(self, xs, zs):
        return tf.pack([multivariate_normal.logpdf(z, self.mu, self.Sigma)
                        for z in tf.unpack(zs)])

ed.set_seed(42)
mu = tf.constant([1.0, 1.0])
Sigma = tf.constant(
[[1.0, 0.1],
 [0.1, 1.0]])
model = Gaussian(mu, Sigma)
variational = ed.MFGaussian(model.num_vars)

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
