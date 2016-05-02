#!/usr/bin/env python
"""
Probability model
    Posterior: (2-dimensional) Bernoulli
Variational model
    Likelihood: Mean-field Bernoulli
"""
import edward as ed
import tensorflow as tf

from edward.stats import bernoulli
from edward.util import get_dims

class Bernoulli:
    """
    p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)
    """
    def __init__(self, p):
        self.p = p
        self.lp = tf.log(p)
        self.num_vars = get_dims(p)[0]

    def log_prob(self, xs, zs):
        # TODO use table lookup for everything not resort to if-elses
        if get_dims(zs)[1] == 1:
            return bernoulli.logpmf(zs[:, 0], p)
        else:
            return tf.concat(0, [self.table_lookup(z) for z in tf.unpack(zs)])

    def table_lookup(self, x):
        elem = self.lp
        for d in range(self.num_vars):
            elem = tf.gather(elem, tf.to_int32(x[d]))
        return elem

ed.set_seed(42)

p = tf.constant(
[[0.4, 0.1],
 [0.1, 0.4]])
model = Bernoulli(p)
variational = ed.MFBernoulli(model.num_vars)

inference = ed.MFVI(model, variational)
inference.run()
