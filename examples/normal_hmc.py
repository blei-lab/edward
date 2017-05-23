#!/usr/bin/env python
"""Correlated normal posterior. Inference with Hamiltonian Monte Carlo.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Empirical, MultivariateNormalTriL

ed.set_seed(42)

# MODEL
z = MultivariateNormalTriL(
    loc=tf.ones(2),
    scale_tril=tf.cholesky(tf.constant([[1.0, 0.8], [0.8, 1.0]])))

# INFERENCE
qz = Empirical(params=tf.Variable(tf.random_normal([1000, 2])))

inference = ed.HMC({z: qz})
inference.run()

# CRITICISM
sess = ed.get_session()
mean, stddev = sess.run([qz.mean(), qz.stddev()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior stddev:")
print(stddev)
