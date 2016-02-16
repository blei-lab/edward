#!/usr/bin/env python
"""
A toy example using alpha-divergence with the reparameterization
gradient.
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

# posterior at N(z; 0, 1)
mu = tf.constant(0.0)
Sigma = tf.constant(1.0)
model = Gaussian(mu, Sigma)
q = bb.MFGaussian(model.num_vars)

# See if it works for initializations roughly 3 std's away.
q.m_unconst = tf.Variable(tf.constant([10.0]))
q.s_unconst = tf.Variable(tf.constant([-10.0]))

inference = bb.AlphaVI(0.5, model, q, n_iter=int(1e6), n_minibatch=1)
inference.run()
