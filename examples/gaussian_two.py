#!/usr/bin/env python
# Probability model
#   Posterior: (2-dimensional) Gaussian
# Variational model
#   Likelihood: Mean-field Gaussian
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
        n_minibatch = get_dims(zs)[0]
        return tf.pack([gaussian_log_prob(zs[m, :], mu, Sigma)
                        for m in range(n_minibatch)])

bb.set_seed(42)

mu = tf.constant([1.0, 1.0])
Sigma = tf.constant(
[[1.0, 0.1],
 [0.1, 1.0]])
model = Gaussian(mu, Sigma)
q = bb.MFGaussian(model.num_vars)

inference = bb.VI(model, q, method="reparam", n_iter=10000)
inference.run()
