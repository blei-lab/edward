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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Variational, Beta
from edward.stats import bernoulli, beta


class BetaBernoulli:
    """p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)"""
    def __init__(self):
        self.n_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs['x'], z))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

    def sample_likelihood(self, zs, n):
        """x | z ~ p(x | z)"""
        out = []
        for s in range(zs.shape[0]):
            out += [{'x': bernoulli.rvs(zs[s, :], size=n).reshape((n,))}]

        return out


ed.set_seed(42)
model = BetaBernoulli()
variational = Variational()
variational.add(Beta(model.n_vars))
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=200)

T = lambda x, z=None: tf.reduce_mean(x['x'])
print(ed.ppc(model, variational, data, T))
