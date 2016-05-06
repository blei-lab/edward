#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Gaussian
Variational model
    Likelihood: Mean-field Gaussian
"""
import edward as ed
import tensorflow as tf

from edward.stats import norm
from edward.variationals import Gaussian
from edward.util import get_dims

class GaussianModel:
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, std)
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.num_vars = 1

    def log_prob(self, xs, zs):
        return tf.pack([norm.logpdf(z, self.mu, self.std)
                        for z in tf.unpack(zs)])

ed.set_seed(42)
mu = tf.constant(1.0)
std = tf.constant(1.0)
model = GaussianModel(mu, std)
variational = Gaussian(model.num_vars)

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
