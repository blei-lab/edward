#!/usr/bin/env python
"""
Probability model
    Posterior: (2-dimensional) Normal
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf

from edward.models import Model, Normal
from edward.stats import multivariate_normal
from edward.util import get_dims

class NormalPosterior:
    """
    p(x, z) = p(z) = p(z | x) = Normal(z; mu, Sigma)
    """
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma
        self.num_vars = get_dims(mu)[0]

    def log_prob(self, xs, zs):
        return multivariate_normal.logpdf(zs, self.mu, self.Sigma)

ed.set_seed(42)
mu = tf.constant([1.0, 1.0])
Sigma = tf.constant(
[[1.0, 0.1],
 [0.1, 1.0]])
model = NormalPosterior(mu, Sigma)
variational = Model()
variational.add(Normal(model.num_vars))

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
