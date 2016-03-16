#!/usr/bin/env python
"""
A simple example from Stan. The model is written in TensorFlow.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
import blackbox as bb
import tensorflow as tf
import numpy as np

from blackbox.stats import bernoulli, beta

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

    def sample_likelihood(self, z, N):
        """x | z ~ p(x | z)"""
        # TODO hard-coded data size
        return bernoulli.rvs(z, size=N)

bb.set_seed(42)
model = BetaBernoulli()
variational = bb.MFBeta(model.num_vars)
data = bb.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = bb.MFVI(model, variational, data)
sess = inference.run(n_iter=200)

T = lambda x: np.mean(x)

# TODO this is waiting the inference refactor, so we can keep the session
print(bb.predictive_check(T, data, model, variational, sess=sess))
