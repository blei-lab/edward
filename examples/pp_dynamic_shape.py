#!/usr/bin/env python
"""Dynamic shapes.

We build a random variable whose size depends on a sample from another
random variable.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Exponential, Dirichlet, Gamma

ed.set_seed(42)

# Prior on scalar hyperparameter to Dirichlet.
alpha = Gamma(alpha=1.0, beta=1.0)

# Prior on size of Dirichlet.
n = 1 + tf.cast(Exponential(lam=0.5), tf.int32)

# Build a vector of ones whose size is n; multiply it by alpha.
p = Dirichlet(alpha=tf.ones([n]) * alpha)

sess = ed.get_session()
print(sess.run(p.value()))
# [ 0.01012419  0.02939712  0.05036638  0.51287931  0.31020424  0.0485355
#   0.0384932 ]
print(sess.run(p.value()))
# [ 0.12836078  0.23335715  0.63828212]
