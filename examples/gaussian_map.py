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
from blackbox.models import PythonModel

class Gaussian(PythonModel):
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, Sigma)
    """
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma
        #self.num_vars = get_dims(mu)[0]
        self.num_vars = 2

    def log_prob(self, xs, zs):
        return tf.pack([norm.logpdf(z, mu, Sigma)
                        for z in tf.unpack(zs)])

bb.set_seed(42)
mu = tf.constant(1.0)
Sigma = tf.constant(1.0)
model = Gaussian(mu, Sigma)
inference = bb.MAP(model)
inference.run(n_iter=1000)
