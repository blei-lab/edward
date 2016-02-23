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
    def __init__(self):
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta_log_prob(zs[:, 0], alpha=1.0, beta=1.0)
        log_lik = tf.pack([
            tf.reduce_sum(bernoulli_log_prob(xs, z)) \
            for z in tf.unpack(zs)])
        return log_lik + log_prior

bb.set_seed(42)
model = BetaBernoulli()
variational = bb.MFBeta(model.num_vars)
data = bb.IID(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = bb.MFVI(model, variational, data)
inference.run(n_iter=10000)
