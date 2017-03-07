#!/usr/bin/env python
"""Dirichlet process.

We sample from a Dirichlet process (with inputted base distribution)
via its stick breaking construction.

References
----------
https://probmods.org/chapters/12-non-parametric-models.html#infinite-discrete-distributions-the-dirichlet-processes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Beta, Normal


def dirichlet_process(alpha, base_cls, sample_n=50, *args, **kwargs):
  """Dirichlet process DP(``alpha``, ``base_cls(*args, **kwargs)``).

  Only works for scalar alpha and scalar base distribution.

  Parameters
  ----------
  alpha : tf.Tensor
    Concentration parameter. Its shape determines the batch shape of the DP.
  base_cls : RandomVariable
    Class of base distribution. Its shape (when instantiated)
    determines the event shape of the DP.
  sample_n : int, optional
    Number of samples for each DP in the batch shape.
  *args, **kwargs : optional
    Arguments passed into ``base_cls``.

  Returns
  -------
  tf.Tensor
    A ``tf.Tensor`` of shape ``[sample_n] + batch_shape + event_shape``,
    where ``sample_n`` is the number of samples for each DP,
    ``batch_shape`` is the number of independent DPs, and
    ``event_shape`` is the shape of the base distribution.
  """
  def cond(k, beta_k, draws, bools):
    # Proceed if at least one bool is True.
    return tf.reduce_any(bools)

  def body(k, beta_k, draws, bools):
    k = k + 1
    beta_k = beta_k * Beta(a=1.0, b=alpha)
    theta_k = base_cls(*args, **kwargs)

    # Assign ongoing samples to the new theta_k.
    indicator = tf.cast(bools, draws.dtype)
    new = indicator * theta_k
    draws = draws * (1.0 - indicator) + new

    flips = tf.cast(Bernoulli(p=beta_k), tf.bool)
    bools = tf.logical_and(flips, tf.equal(draws, theta_k))
    return k, beta_k, draws, bools

  k = 0
  beta_k = Beta(a=tf.ones(sample_n), b=alpha * tf.ones(sample_n))
  theta_k = base_cls(*args, **kwargs)

  # Initialize all samples as theta_k.
  draws = tf.ones(sample_n) * theta_k
  # Flip ``sample_n`` coins, one for each sample.
  flips = tf.cast(Bernoulli(p=beta_k), tf.bool)
  # Get boolean tensor for samples that return heads
  # and are currently equal to theta_k.
  bools = tf.logical_and(flips, tf.equal(draws, theta_k))

  total_sticks, _, samples, _ = tf.while_loop(
      cond, body, loop_vars=[k, beta_k, draws, bools])
  return total_sticks, samples


dp = dirichlet_process(0.1, Normal, mu=0.0, sigma=1.0)
sess = tf.Session()
print(sess.run(dp))
print(sess.run(dp))
print(sess.run(dp))
