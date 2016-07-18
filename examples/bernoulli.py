#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Bernoulli
Variational model
    Likelihood: Mean-field Bernoulli
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Variational, Bernoulli
from edward.stats import bernoulli


class BernoulliPosterior:
    """p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)"""
    def __init__(self, p):
        self.p = p

    def log_prob(self, xs, zs):
        return bernoulli.logpmf(zs, p)


ed.set_seed(42)
p = tf.constant(0.6)
model = BernoulliPosterior(p)
variational = Variational()
variational.add(Bernoulli())

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
