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
p = Beta(a=1.0, b=1.0)
x = Bernoulli(p=tf.ones(10) * p)

# INFERENCE
qp = Empirical(params=tf.Variable(tf.zeros([1000]) + 0.5))

proposal_p = Beta(a=3.0, b=9.0)

inference = ed.MetropolisHastings({p: qp}, {p: proposal_p}, data={x: x_data})
inference.run()

# CRITICISM
# exact posterior has mean 0.25 and std 0.12
sess = ed.get_session()
mean, std = sess.run([qp.mean(), qp.std()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior std:")
print(std)
