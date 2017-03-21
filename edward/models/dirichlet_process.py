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
    """Dirichlet process :math:`\mathcal{DP}(\\alpha, H)`.

    It has two parameters: a positive real value :math:`\\alpha`,
    known as the concentration parameter (``alpha``), and a base
    distribution :math:`H` (``base_cls(*args, **kwargs)``).

    Parameters
    ----------
    alpha : tf.Tensor
      Concentration parameter. Must be positive real-valued. Its shape
      determines the number of independent DPs (batch shape).
    base_cls : RandomVariable
      Class of base distribution. Its shape (when instantiated)
      determines the shape of an individual DP (event shape).
    *args, **kwargs : optional
      Arguments passed into ``base_cls``.

    Examples
    --------
    >>> # scalar concentration parameter, scalar base distribution
    >>> dp = DirichletProcess(0.1, Normal, mu=0.0, sigma=1.0)
    >>> assert dp.get_shape() == ()
    >>>
    >>> # vector of concentration parameters, matrix of Exponentials
    >>> dp = DirichletProcess(tf.constant([0.1, 0.4]),
    ...                       Exponential, lam=tf.ones([5, 3]))
    >>> assert dp.get_shape() == (2, 5, 3)
    """
    parameters = locals()
    parameters.pop("self")
    with tf.name_scope(name, values=[alpha]) as ns:
      with tf.control_dependencies([]):
        self._alpha = tf.identity(alpha, name="alpha")
        self._base_cls = base_cls
        self._base_args = args
        self._base_kwargs = kwargs

        # Instantiate base distribution.
        self._base = self._base_cls(*self._base_args, **self._base_kwargs)
        # Define atoms of Dirichlet process, storing only the first as default.
        self._theta = tf.expand_dims(
            self._base.sample(self.get_batch_shape()), 0)

        super(DirichletProcess, self).__init__(
            dtype=tf.int32,
            is_continuous=False,
            is_reparameterized=False,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._alpha, self._theta],
            name=ns,
            value=value)

  @property
  def alpha(self):
    """Concentration parameter."""
    return self._alpha

  @property
  def theta(self):
    """Atoms. It has shape [None] + batch_shape + event_shape, where
    the first dimension is the number of atoms, instantiated only as
    needed."""
    return self._theta

  def _batch_shape(self):
    return tf.shape(self.alpha)

  def _get_batch_shape(self):
    return self.alpha.get_shape()

  def _event_shape(self):
    return tf.shape(self._base)

  def _get_event_shape(self):
    return self._base.get_shape()

  def _sample_n(self, n, seed=None):
    """Sample ``n`` draws from the DP. Draws from the base
    distribution are memoized across ``n`` and across calls to
    ``sample()``.

    Draws from the base distribution are not memoized across the batch
    shape, i.e., each independent DP in the batch shape has its own
    memoized samples.

    Returns
    -------
    tf.Tensor
      A ``tf.Tensor`` of shape ``[n] + batch_shape + event_shape``,
      where ``n`` is the number of samples for each DP,
      ``batch_shape`` is the number of independent DPs, and
      ``event_shape`` is the shape of the base distribution.

    Notes
    -----
    The implementation has one inefficiency, which is that it draws
    (batch_shape,) samples from the base distribution when adding a
    new persistent state. Ideally, we would only draw new samples for
    those in the loop which require it.
    """
    if seed is not None:
      raise NotImplementedError("seed is not implemented.")

    batch_shape = self.get_batch_shape().as_list()
    event_shape = self.get_event_shape().as_list()
    rank = 1 + len(batch_shape) + len(event_shape)
    # Note this is for scoping within the while loop's body function.
    self._temp_scope = [n, batch_shape, event_shape, rank]

    k = tf.constant(0)

    # Draw stick probability, then flip coin; perform this for each
    # sample and each DP in the batch shape. It has shape (n, batch_shape).
    beta_k = Beta(a=tf.ones_like(self.alpha), b=self.alpha).sample(n)
    flips = Bernoulli(p=beta_k)
    # Define boolean tensor. It is True for samples that require continuing
    # the while loop and False for samples that can receive their base
    # distribution (coin lands heads).
    bools = tf.cast(1 - flips, tf.bool)

    # Extract base distribution. It has shape (batch_shape, event_shape).
    theta = self.theta
    theta_k = tf.gather(theta, k)

    # Initialize all samples as the first base distribution.
    draws = tf.tile(tf.expand_dims(theta_k, 0), [n] + [1] * (rank - 1))

    theta_shape = tf.TensorShape([None])
    if len(theta.shape) > 1:
      theta_shape = theta_shape.concatenate(theta.shape[1:])

    _, _, self._theta, samples = tf.while_loop(
        self._sample_n_cond, self._sample_n_body,
        loop_vars=[k, bools, theta, draws],
        shape_invariants=[k.shape, bools.shape, theta_shape, draws.shape])

    return samples

  def _sample_n_cond(self, k, bools, theta, draws):
    # Proceed if at least one bool is True.
    return tf.reduce_any(bools)

  def _sample_n_body(self, k, bools, theta, draws):
    n, batch_shape, event_shape, rank = self._temp_scope
    k += 1

    # If necessary, add a new persistent state to theta.
    theta = tf.cond(
        tf.shape(theta)[0] - 1 >= k,
        lambda: theta,
        lambda: tf.concat(
            [theta, tf.expand_dims(self._base.sample(batch_shape), 0)], 0))
    theta_k = tf.gather(theta, k)

    # Assign True samples to the new theta_k.
    if len(bools.get_shape()) <= 1:
      bools_tile = bools
    else:
      # ``tf.where`` only index subsets when ``bools`` is at most a
      # vector. In general, ``bools`` has shape (n, batch_shape).
      # Therefore we tile ``bools`` to be of shape
      # (n, batch_shape, event_shape) in order to index per-element.
      bools_tile = tf.tile(tf.reshape(
          bools, [n] + batch_shape + [1] * len(event_shape)),
          [1] + [1] * len(batch_shape) + event_shape)

    theta_k_tile = tf.tile(tf.expand_dims(theta_k, 0), [n] + [1] * (rank - 1))
    draws = tf.where(bools_tile, theta_k_tile, draws)

    # Draw new stick probability, then flip coin.
    beta_k = Beta(a=tf.ones_like(self.alpha), b=self.alpha).sample(n)
    flips = Bernoulli(p=beta_k)
    # If coin lands heads, assign sample's corresponding bool to False
    # (this ends its "while loop").
    bools = tf.where(tf.cast(flips, tf.bool), tf.zeros_like(bools), bools)
    return k, bools, theta, draws
