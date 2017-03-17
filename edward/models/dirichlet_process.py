from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models.random_variables import Bernoulli, Beta
from tensorflow.contrib.distributions import Distribution


class DirichletProcess(RandomVariable, Distribution):
  def __init__(self, alpha, base_cls, validate_args=False, allow_nan_stats=True,
               name="DirichletProcess", value=None, *args, **kwargs):
    """Dirichlet process DP(``alpha``, ``base_cls(*args, **kwargs)``).

    Parameters
    ----------
    alpha : tf.Tensor
      Concentration parameter. Its shape determines the batch shape of the DP.
    base_cls : RandomVariable
      Class of base distribution. Its shape (when instantiated)
      determines the event shape of the DP.
    *args, **kwargs : optional
      Arguments passed into ``base_cls``.

    Examples
    --------
    >>> # scalar concentration parameter, scalar base distribution
    >>> dp = DirichletProcess(0.1, Normal, mu=0.0, sigma=1.0)
    >>> dp.get_shape() == ()
    True
    >>>
    >>> # vector of concentration parameters, matrix of Exponentials
    >>> dp = DirichletProcess(tf.constant([0.1, 0.4]),
    ...                       Exponential, lam=tf.ones([5, 3]))
    >>> dp.get_shape() == (2, 5, 3)
    True
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
    """Sample ``n`` draws from the DP. Draws from the base
    distribution are memoized across ``n``.

    Draws from the base distribution are not memoized across the batch
    shape: i.e., each independent DP in the batch shape has its own
    memoized samples.  Similarly, draws are not memoized across calls
    to ``sample()``.

    Returns
    -------
    tf.Tensor
      A ``tf.Tensor`` of shape ``[n] + batch_shape + event_shape``,
      where ``n`` is the number of samples for each DP,
      ``batch_shape`` is the number of independent DPs, and
      ``event_shape`` is the shape of the base distribution.

    Notes
    -----
    The implementation has only one inefficiency, which is that it
    draws (n, batch_shape) samples from the base distribution at each
    iteration of the while loop. Ideally, we would only draw new
    samples for those in the loop returning True.
    """
    if seed is not None:
      raise NotImplementedError("seed is not implemented.")

    batch_shape = self._get_batch_shape().as_list()
    event_shape = self._get_event_shape().as_list()
    rank = 1 + len(batch_shape) + len(event_shape)
    # Note this is for scoping within the while loop's body function.
    self._temp_scope = [n, batch_shape, event_shape, rank]

    # First stick index.
    k = 0
    # First stick probability, one for each sample and each DP in the
    # batch shape. It has shape (n, batch_shape).
    beta_k = Beta(a=tf.ones_like(self._alpha), b=self._alpha).sample(n)
    # First base distribution.
    # It has shape (n, batch_shape, event_shape).
    theta_k = tf.tile(  # make (batch_shape, event_shape), then memoize across n
        tf.expand_dims(self._base_cls(*self._base_args, **self._base_kwargs).
                       sample(batch_shape), 0),
        [n] + [1] * (rank - 1))

    # Initialize all samples as the first base distribution.
    draws = theta_k
    # Flip coins for each stick probability.
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
    n, batch_shape, event_shape, rank = self._temp_scope

    k += 1
    beta_k *= Beta(a=tf.ones_like(self._alpha), b=self._alpha).sample(n)
    theta_k = tf.tile(  # make (batch_shape, event_shape), then memoize across n
        tf.expand_dims(self._base_cls(*self._base_args, **self._base_kwargs).
                       sample(batch_shape), 0),
        [n] + [1] * (rank - 1))

    if len(bools.get_shape()) > 1:
      # ``tf.where`` only index subsets when ``bools`` is at most a
      # vector. In general, ``bools`` has shape (n, batch_shape).
      # Therefore we tile ``bools`` to be of shape
      # (n, batch_shape, event_shape) in order to index per-element.
      bools = tf.tile(tf.reshape(
          bools, [n] + batch_shape + [1] * len(event_shape)),
          [1] + [1] * len(batch_shape) + event_shape)

    # Assign True samples to the new theta_k.
    draws = tf.where(bools, theta_k, draws)

    flips = Bernoulli(p=beta_k)
    bools = tf.logical_and(
        tf.cast(flips, tf.bool),
        tf.reduce_all(tf.equal(draws, theta_k),  # reduce event_shape
                      [i for i in range(1 + len(batch_shape), rank)]))
    return k, beta_k, draws, bools
