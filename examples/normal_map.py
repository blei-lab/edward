#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Normal
Inference: Maximum a posteriori
"""
import edward as ed
import tensorflow as tf

from edward.stats import norm

class NormalModel:
    """
    p(x, z) = Normal(x; z, std) Normal(z; mu, std)
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = norm.logpdf(zs, self.mu, self.std)
        log_lik = tf.pack([tf.reduce_sum(norm.logpdf(xs, z, self.std))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

ed.set_seed(42)
mu = tf.constant(3.0)
std = tf.constant(0.1)
model = NormalModel(mu, std)
data = ed.Data(tf.constant((3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = ed.MAP(model, data)
inference.run(n_iter=200, n_print=50)
