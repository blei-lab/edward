#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Normal
Variational model
    Likelihood: Mean-field Normal
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Variational, Normal
from edward.stats import norm


class NormalPosterior:
    """p(x, z) = p(z) = p(z | x) = Normal(z; mu, std)"""
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def log_prob(self, xs, zs):
        return norm.logpdf(zs, self.mu, self.std)


ed.set_seed(42)
mu = tf.constant(1.0)
std = tf.constant(1.0)
model = NormalPosterior(mu, std)
variational = Variational()
variational.add(Normal())

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
