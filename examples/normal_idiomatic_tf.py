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
import edward as ed
import tensorflow as tf

from edward.stats import norm
from edward.variationals import Variational, Normal

class NormalPosterior:
    """
    p(x, z) = p(z) = p(z | x) = Normal(z; mu, std)
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.num_vars = 1

    def log_prob(self, xs, zs):
        return norm.logpdf(zs, self.mu, self.std)

ed.set_seed(42)
mu = tf.constant(1.0)
std = tf.constant(1.0)
model = NormalPosterior(mu, std)
variational = Variational()
variational.add(Normal(model.num_vars))

inference = ed.MFVI(model, variational)
sess = inference.initialize()
for t in range(1000):
    loss = inference.update(sess)
    inference.print_progress(t, loss, sess)
