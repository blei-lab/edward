#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Gaussian
Variational model
    Likelihood: Mean-field Gaussian
"""
import tensorflow as tf
import edward as ed

from edward.stats import norm
from edward.util import get_dims

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

ed.set_seed(42)
mu = tf.constant(1.0)
std = tf.constant(1.0)
model = Gaussian(mu, std)
variational = ed.MFGaussian(model.num_vars)

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
