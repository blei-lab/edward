#!/usr/bin/env python
"""
A simple example from Stan. The model is written in TensorFlow.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
import tensorflow as tf
import blackbox as bb

from blackbox.stats import bernoulli_log_prob, beta_log_prob

class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
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
data = tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32)
model = BetaBernoulli(data)
variational = bb.MFBeta(model.num_vars)

inference = bb.MFVI(model, variational)
inference.run(n_iter=10000)
