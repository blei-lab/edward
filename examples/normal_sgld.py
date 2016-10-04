#!/usr/bin/env python
"""Correlated normal posterior. Inference with stochastic gradient
Langevin dynamics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Empirical, MultivariateNormalFull

ed.set_seed(42)

# MODEL
z = MultivariateNormalFull(
    mu=tf.ones(2),
    sigma=tf.constant([[1.0, 0.8], [0.8, 1.0]]))

# INFERENCE
qz = Empirical(params=tf.Variable(tf.random_normal([2000, 2])))

inference = ed.SGLD({z: qz})
inference.run(step_size=5.0)

# CRITICISM
sess = ed.get_session()
mean, std = sess.run([qz.mean(), qz.std()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior std:")
print(std)
