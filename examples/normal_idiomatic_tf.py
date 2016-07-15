#!/usr/bin/env python
"""
This demonstrates a more idiomatic TensorFlow example. Instead of
running inference.run(), we may want direct access to the TensorFlow
session and to manipulate various objects during inference.

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
inference.initialize()
for t in range(1000):
    loss = inference.update()
    inference.print_progress(t, loss)
