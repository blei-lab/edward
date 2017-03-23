#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical

ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
p = Beta(1.0, 1.0)
x = Bernoulli(tf.ones(10) * p)

# INFERENCE
qp = Empirical(params=tf.Variable(tf.zeros([1000]) + 0.5))

proposal_p = Beta(3.0, 9.0)

inference = ed.MetropolisHastings({p: qp}, {p: proposal_p}, data={x: x_data})
inference.run()

# CRITICISM
# exact posterior has mean 0.25 and std 0.12
sess = ed.get_session()
mean, stddev = sess.run([qp.mean(), qp.stddev()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior stddev:")
print(stddev)
