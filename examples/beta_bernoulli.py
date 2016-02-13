#!/usr/bin/env python
# A simple example from Stan.
# Probability model
#   Prior: Beta
#   Likelihood: Bernoulli
# Variational model
#   Likelihood: Mean-field Beta
import numpy as np
import tensorflow as tf
import blackbox as bb

from blackbox.stats import bernoulli_log_prob, beta_log_prob

class BernoulliModel:
    """
    p(z) = Beta(z; 1, 1)
    p(x|z) = Bernoulli(x; z)
    """
    def __init__(self, data):
        self.data = data
        self.num_vars = 1

    def log_prob(self, zs):
        log_prior = beta_log_prob(zs[:, 0], alpha=1.0, beta=1.0)
        log_lik = tf.pack([
            tf.reduce_sum(bernoulli_log_prob(self.data, z)) \
            for z in tf.unpack(zs)])
        return log_lik + log_prior

bb.set_seed(42)

data = tf.constant((0,1,0,0,0,0,0,0,0,1), dtype=tf.float32)
model = BernoulliModel(data)
q = bb.MFBeta(model.num_vars)

inference = bb.VI(model, q, n_minibatch=5)
inference.run()
