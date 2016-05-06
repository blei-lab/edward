#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Bernoulli
Variational model
    Likelihood: Mean-field Bernoulli
"""
import edward as ed
import tensorflow as tf

from edward.stats import bernoulli
from edward.variationals import Bernoulli

class BernoulliModel:
    """
    p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)
    """
    def __init__(self, p):
        self.p = p

    def log_prob(self, xs, zs):
        return bernoulli.logpmf(zs, p)

ed.set_seed(42)
p = tf.constant(0.6)
model = BernoulliModel(p)
variational = Bernoulli(1)

inference = ed.MFVI(model, variational)
inference.run()
