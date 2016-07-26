#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Normal
Inference: Maximum a posteriori
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.stats import norm


class NormalModel:
    """p(x, z) = Normal(x; z, std) Normal(z; mu, std)"""
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.n_vars = 1

    def log_prob(self, xs, zs):
        log_prior = norm.logpdf(zs, self.mu, self.std)
        log_lik = tf.pack([tf.reduce_sum(norm.logpdf(xs['x'], z, self.std))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior


ed.set_seed(42)
mu = tf.constant(3.0)
std = tf.constant(0.1)
model = NormalModel(mu, std)
data = {'x': np.array([3]*20 + [0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)}

inference = ed.MAP(model, data)
inference.run(n_iter=200, n_print=50)
