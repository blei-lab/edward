#!/usr/bin/env python
"""
A simple example from Stan. The model is written in TensorFlow.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Inference: Maximum a posteriori
"""
import edward as ed
import tensorflow as tf

from edward.stats import bernoulli, beta

class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs, z)) \
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

ed.set_seed(42)
model = BetaBernoulli()
data = ed.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = ed.MAP(model, data, transform=tf.sigmoid)
inference.run(n_iter=100, n_print=10)
