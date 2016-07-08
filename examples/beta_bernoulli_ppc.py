#!/usr/bin/env python
"""
A simple coin flipping example. The model is written in TensorFlow.
Inspired by Stan's toy example.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Variational, Beta
from edward.stats import bernoulli, beta

class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs, z))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

    def sample_likelihood(self, zs, size):
        """x | z ~ p(x | z)"""
        out = np.zeros((zs.shape[0], size))
        for s in range(zs.shape[0]):
            out[s,:] = bernoulli.rvs(zs[s,:], size=size).reshape((size,))

        return out

ed.set_seed(42)
model = BetaBernoulli()
variational = Variational()
variational.add(Beta(model.num_vars))
data = ed.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=200)

T = lambda y, z=None: tf.reduce_mean(y)
print(ed.ppc(model, variational, data, T))
