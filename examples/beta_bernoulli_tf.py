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
import edward as ed

from edward.stats import bernoulli, beta

class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs[:, 0], a=1.0, b=1.0)
        log_lik = tf.concat(0, [
            tf.reduce_sum(bernoulli.logpmf(xs, z)) \
            for z in tf.unpack(zs)])
        return log_lik + log_prior

ed.set_seed(42)
model = BetaBernoulli()
variational = ed.MFBeta(model.num_vars)
data = ed.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=10000)
