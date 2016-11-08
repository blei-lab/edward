#!/usr/bin/env python
"""Dirichlet process.

We sample from a Dirichlet process (with no base distribution) by
using its stick breaking construction.

References
----------
https://probmods.org/non-parametric-models.html#infinite-discrete-distributions-the-dirichlet-processes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Beta


def dirichlet_process(alpha):
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


dp = dirichlet_process(alpha=1.0)

sess = tf.Session()
print(sess.run(dp))
