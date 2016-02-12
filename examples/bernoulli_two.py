#!/usr/bin/env python
# Probability model
#   Posterior: 2-dimensional Bernoulli
# Variational model
#   Likelihood: Mean-field Bernoulli
import numpy as np
import tensorflow as tf

import blackbox as bb
from blackbox.dists import bernoulli_log_prob
from blackbox.likelihoods import MFBernoulli

class Bernoulli:
    """
    p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)
    """
    def __init__(self, p):
        self.p = p
        #self.num_vars = tf.rank(p)
        self.num_vars = 2 # TODO hardcoded

    def log_prob(self, zs):
        # TODO this breaks for dim(z) = 1
        out = tf.pack([self.table_lookup(z) for z in tf.unpack(zs)])
        return out

    def table_lookup(self, x):
        elem = tf.log(p)
        for d in range(self.num_vars):
            elem = tf.gather(elem, tf.to_int32(x[d]))
        return elem

np.random.seed(42)
tf.set_random_seed(42)

p = tf.constant(
[[0.4, 0.1],
 [0.1, 0.4]])
model = Bernoulli(p)
q = MFBernoulli(2) # TODO hardcoded

inference = bb.VI(model, q, n_minibatch=100)
inference.run()
