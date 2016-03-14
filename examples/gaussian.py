#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Gaussian
Variational model
    Likelihood: Mean-field Gaussian
"""
import tensorflow as tf
import blackbox as bb

from blackbox.stats import norm
from blackbox.util import get_dims

class Gaussian:
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, std)
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.num_vars = 1

    def log_prob(self, xs, zs):
        return tf.concat(0, [norm.logpdf(z, self.mu, self.std)
                         for z in tf.unpack(zs)])

bb.set_seed(42)
mu = tf.constant(1.0)
std = tf.constant(1.0)
model = Gaussian(mu, std)
variational = bb.MFGaussian(model.num_vars)

inference = bb.MFVI(model, variational)
inference.run(n_iter=10000)
