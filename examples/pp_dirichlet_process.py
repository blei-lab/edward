#!/usr/bin/env python
"""Dirichlet process.

We implement sample generation from a Dirichlet process (with no base
distribution) via its stick breaking construction. It is a streamlined
implementation of the ``DirichletProcess`` random variable in Edward.

References
----------
https://probmods.org/chapters/12-non-parametric-models.html#infinite-discrete-distributions-the-dirichlet-processes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Beta, DirichletProcess, Normal


def dirichlet_process(alpha):
  """Demo of stochastic while loop for stick breaking construction."""
  def cond(k, beta_k):
    flip = Bernoulli(p=beta_k)
    return tf.equal(flip, tf.constant(1))

  def body(k, beta_k):
    beta_k = beta_k * Beta(a=1.0, b=alpha)
    return k + 1, beta_k

  k = tf.constant(0)
  beta_k = Beta(a=1.0, b=alpha)
  stick_num, stick_beta = tf.while_loop(cond, body, loop_vars=[k, beta_k])
  return stick_num


dp = dirichlet_process(alpha=0.1)

# The number of sticks broken is dynamic, changing across evaluations.
sess = tf.Session()
print(sess.run(dp))
print(sess.run(dp))

# Demo of the DirichletProcess random variable in Edward.
# It is associated to a sample tensor, which in turn is associated to
# one of its atoms (base distributions).
base_cls = Normal
kwargs = {'mu': 0.0, 'sigma': 1.0}
dp = DirichletProcess(0.1, base_cls, **kwargs)
print(dp)

# ``theta`` is the distribution indirectly returned by the DP.
theta = base_cls(value=tf.cast(dp, tf.float32), **kwargs)
print(theta)

# Fetching theta is the same as fetching the Dirichlet process.
sess = tf.Session()
print(sess.run([dp, theta]))
print(sess.run([dp, theta]))

# DirichletProcess can also take in non-scalar concentrations and bases.
base_cls = Exponential
kwargs = {'lam': tf.ones([5, 2])}
dp = DirichletProcess(tf.constant([0.1, 0.6, 0.4]), base_cls, **kwargs)
print(dp)
