#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Gaussian
Variational model
    Likelihood: Mean-field Gaussian
"""
import numpy as np
import tensorflow as tf
import blackbox as bb

from blackbox.stats import gaussian_log_prob
from blackbox.util import get_dims

class Gaussian:
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, Sigma)
    """
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma
        self.num_vars = get_dims(mu)[0]

    def log_prob(self, zs):
        return tf.pack([gaussian_log_prob(z, mu, Sigma)
                        for z in tf.unpack(zs)])

bb.set_seed(42)

mu = tf.constant(1.0)
Sigma = tf.constant(1.0)
model = Gaussian(mu, Sigma)
q = bb.MFGaussian(model.num_vars)

#q.m_unconst = tf.Variable(tf.constant([20.0]))
#q.s_unconst = tf.Variable(tf.constant([0.0]))

inference = bb.MFVI(model, q, n_iter=10000)
inference.run()
