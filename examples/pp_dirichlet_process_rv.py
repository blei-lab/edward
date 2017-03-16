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

from edward.models import Bernoulli, Beta, Exponential, Normal, RandomVariable
from tensorflow.contrib.distributions import Distribution


class DirichletProcess(RandomVariable, Distribution):
  def __init__(self, alpha, base_cls, validate_args=False, allow_nan_stats=True,
               name="DirichletProcess", value=None, *args, **kwargs):
    """Dirichlet process DP(``alpha``, ``base_cls(*args, **kwargs)``).

    Only works for scalar ``alpha``;  the base distribution can have
    arbitrary dimensions.

    Parameters
    ----------
    alpha : tf.Tensor
      Concentration parameter. Its shape determines the batch shape of the DP.
    base_cls : RandomVariable
      Class of base distribution. Its shape (when instantiated)
      determines the event shape of the DP.
    *args, **kwargs : optional
      Arguments passed into ``base_cls``.
    """
    with tf.name_scope(name, values=[alpha]) as ns:
      with tf.control_dependencies([]):
        self._alpha = tf.identity(alpha, name="alpha")
        self._base_cls = base_cls
        self._base_args = args
        self._base_kwargs = kwargs

        # Instantiate base for use in other methods such as `_get_event_shape`.
        self._base = self._base_cls(*self._base_args, **self._base_kwargs)

        super(DirichletProcess, self).__init__(
            dtype=tf.int32,
            parameters={"alpha": self._alpha,
                        "base_cls": self._base_cls,
                        "args": self._base_args,
                        "kwargs": self._base_kwargs},
            is_continuous=False,
            is_reparameterized=False,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns,
            value=value)

  @property
  def alpha(self):
    """Concentration parameter."""
    return self._alpha

  def _batch_shape(self):
    return tf.convert_to_tensor(self.get_batch_shape())

  def _get_batch_shape(self):
    return self._alpha.get_shape()

  def _event_shape(self):
    return tf.convert_to_tensor(self.get_event_shape())

  def _get_event_shape(self):
    return self._base.get_shape()

  def _sample_n(self, n, seed=None):
    """
    Returns
    -------
    tf.Tensor
      A ``tf.Tensor`` of shape ``[n] + batch_shape + event_shape``,
      where ``n`` is the number of samples for each DP,
      ``batch_shape`` is the number of independent DPs, and
      ``event_shape`` is the shape of the base distribution.

    Notes
    -----
    The only inefficiency is in drawing (n, batch_shape) samples from
    the base distribution at each iteration of the while loop. Ideally,
    we would only draw new samples for those in the loop returning True.
    """
    if seed is not None:
      raise NotImplementedError("seed is not implemented.")

    batch_shape = self._get_batch_shape().as_list()
    event_shape = self._get_event_shape().as_list()
    rank = 1 + len(batch_shape) + len(event_shape)
    # Note this is for scoping within the while loop's body function.
    self._temp_scope = [n, batch_shape, rank]

    k = 0
    beta_k = Beta(a=tf.ones_like(self._alpha), b=self._alpha).sample(n)
    theta_k = self._base_cls(*self._base_args, **self._base_kwargs).sample(
        [n] + batch_shape)

    # Initialize all samples as theta_k.
    # It has shape (n, batch_shape, event_shape).
    draws = theta_k
    # Flip coins, one for each sample and each DP in the batch shape.
    # It has shape (n, batch_shape).
    flips = Bernoulli(p=beta_k)
    # Get boolean tensor, returning True for samples that return heads
    # and are currently equal to theta_k.
    # It has shape (n, batch_shape).
    bools = tf.logical_and(
        tf.cast(flips, tf.bool),
        tf.reduce_all(tf.equal(draws, theta_k),  # reduce event_shape
                      [i for i in range(1 + len(batch_shape), rank)]))

    _, _, samples, _ = tf.while_loop(
        self._sample_n_cond, self._sample_n_body,
        loop_vars=[k, beta_k, draws, bools])
    return samples

  def _sample_n_cond(self, k, beta_k, draws, bools):
    # Proceed if at least one bool is True.
    return tf.reduce_any(bools)

  def _sample_n_body(self, k, beta_k, draws, bools):
    n, batch_shape, rank = self._temp_scope

    k += 1
    beta_k *= Beta(a=tf.ones_like(self._alpha), b=self._alpha).sample(n)
    theta_k = self._base_cls(*self._base_args, **self._base_kwargs).sample(
        [n] + batch_shape)

    # Assign True samples to the new theta_k.
    # Note ``tf.where`` only works if ``bools`` is at most a vector.
    # Since ``bools`` has shape (n, batch_shape), it only works
    # with scalar batch shape.
    # TODO broadcast bools
    draws = tf.where(bools, theta_k, draws)

    flips = Bernoulli(p=beta_k)
    bools = tf.logical_and(
        tf.cast(flips, tf.bool),
        tf.reduce_all(tf.equal(draws, theta_k),  # reduce event_shape
                      [i for i in range(1 + len(batch_shape), rank)]))
    return k, beta_k, draws, bools


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

# This also works for non-scalar base distributions.
base_cls = Exponential
kwargs = {'lam': tf.ones([5, 2])}
dp = DirichletProcess(0.1, base_cls, **kwargs)
print(dp)
