#!/usr/bin/env python
"""
A simple example from Stan. The model is written in TensorFlow.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
"""
import tensorflow as tf
import blackbox as bb

from blackbox.stats import bernoulli, beta

class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def get_num_vars(self, xs=None):
        # This returns the number of parameters for MAP to optimize.
        return 2

    def log_prob(self, xs, zs):
        # TODO
        # zs is a n_minibatch x 2 matrix, where each column represents
        # the shape and scale parameters to be optimized
        # But for it to be plugged into this prior,
        # zs must be a n_minibatch x 1 matrix, where the column represents
        # the latent variable constrained in [0,1]
        log_prior = beta.logpdf(zs[0], a=1.0, b=1.0)
        log_lik = tf.pack([
            tf.reduce_sum(bernoulli.logpmf(xs, z)) \
            for z in tf.unpack(zs)])
        return log_lik + log_prior

bb.set_seed(42)
model = BetaBernoulli()
#variational = bb.MFBeta(model.num_vars)
data = bb.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32))

inference = bb.MAP(model, data)
inference.run(n_iter=10000)
