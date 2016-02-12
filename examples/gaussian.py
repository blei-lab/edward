#!/usr/bin/env python
# Probability model
#   Posterior: (1-dimensional) Gaussian
# Variational model
#   Likelihood: Mean-field Gaussian
# TODO
# enable learning Sigma
# minibatches
#   gaussian_log_prob: enable minibatches
#   MFGaussian
#   gaussian posterior
# higher dimensions
# reparameterization trick
# gaussian entropy
import numpy as np
import tensorflow as tf
import blackbox as bb

from blackbox.likelihoods import MFGaussian
from blackbox.dists import gaussian_log_prob
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
        # TODO generalize to larger minibatch
        return gaussian_log_prob(zs[0, :], mu, Sigma)

bb.set_seed(42)

mu = tf.constant(1.0)
Sigma = tf.constant(1.0)
model = Gaussian(mu, Sigma)
q = MFGaussian(model.num_vars)

inference = bb.VI(model, q, n_iter=1000, n_minibatch=1)
inference.run()
