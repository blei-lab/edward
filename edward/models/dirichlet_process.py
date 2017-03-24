from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution

try:
  from edward.models.random_variables import Bernoulli, Beta
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class DirichletProcess(RandomVariable, Distribution):
  """Dirichlet process :math:`\mathcal{DP}(\\alpha, H)`.

  It has two parameters: a positive real value :math:`\\alpha`,
  known as the concentration parameter (``alpha``), and a base
  distribution :math:`H` (``base``).
  """
  def __init__(self, alpha, base, validate_args=False, allow_nan_stats=True,
               name="DirichletProcess", *args, **kwargs):
    """Initialize a batch of Dirichlet processes.

    Parameters
    ----------
    alpha : tf.Tensor
      Concentration parameter. Must be positive real-valued. Its shape
      determines the number of independent DPs (batch shape).
    base : RandomVariable
      Base distribution. Its shape determines the shape of an
      individual DP (event shape).

    Examples
    --------
    >>> # scalar concentration parameter, scalar base distribution
    >>> dp = DirichletProcess(0.1, Normal(mu=0.0, sigma=1.0))
    >>> assert dp.shape == ()
    >>>
    >>> # vector of concentration parameters, matrix of Exponentials
    >>> dp = DirichletProcess(tf.constant([0.1, 0.4]),
    ...                       Exponential(lam=tf.ones([5, 3])))
    >>> assert dp.shape == (2, 5, 3)
    """
    parameters = locals()
    parameters.pop("self")
    with tf.name_scope(name, values=[alpha]) as ns:
      with tf.control_dependencies([
          tf.assert_positive(alpha),
      ] if validate_args else []):
        if validate_args and isinstance(base, RandomVariable):
          raise TypeError("base must be a ed.RandomVariable object.")

        self._alpha = tf.identity(alpha, name="alpha")
        self._base = base

        # Create empty tensor to store future atoms.
        self._theta = tf.zeros(
            [0] +
            self.get_batch_shape().as_list() +
            self.get_event_shape().as_list(),
            dtype=self._base.dtype)

        # Instantiate beta distribution for stick breaking proportions.
        self._betadist = Beta(a=tf.ones_like(self._alpha), b=self._alpha)
        # Create empty tensor to store stick breaking proportions.
        self._beta = tf.zeros(
            [0] + self.get_batch_shape().as_list(),
            dtype=self._betadist.dtype)

      super(DirichletProcess, self).__init__(
          dtype=tf.int32,
          is_continuous=False,
          is_reparameterized=False,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._alpha, self._beta, self._theta],
          name=ns,
          *args, **kwargs)

  @property
  def alpha(self):
    """Concentration parameter."""
    return self._alpha

  @property
  def base(self):
    """Base distribution used for drawing the atoms."""
    return self._base

  @property
  def beta(self):
    """Stick breaking proportions. It has shape [None] + batch_shape, where
    the first dimension is the number of atoms, instantiated only as
    needed."""
    return self._beta

  @property
  def theta(self):
    """Atoms. It has shape [None] + batch_shape + event_shape, where
    the first dimension is the number of atoms, instantiated only as
    needed."""
    return self._theta

  def _batch_shape(self):
    return tf.shape(self.alpha)

  def _get_batch_shape(self):
    return self.alpha.shape

  def _event_shape(self):
    return tf.shape(self.base)

  def _get_event_shape(self):
    return self.base.shape

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

    # Start at the beginning of the stick, i.e. the k'th index
    k = tf.constant(0)

    # Define boolean tensor. It is True for samples that require continuing
    # the while loop and False for samples that can receive their base
    # distribution (coin lands heads). Also note that we need one bool for
    # each sample
    bools = tf.ones([n] + batch_shape, dtype=tf.bool)

    # Initialize all samples as zero, they will be overwritten in any case
    draws = tf.zeros([n] + batch_shape + event_shape, dtype=self.base.dtype)

    # Calculate shape invariance conditions for theta and beta as these
    # can change shape between loop iterations.
    theta_shape = tf.TensorShape([None])
    beta_shape = tf.TensorShape([None])
    if len(self.theta.shape) > 1:
      theta_shape = theta_shape.concatenate(self.theta.shape[1:])
      beta_shape = beta_shape.concatenate(self.beta.shape[1:])

    # While we have not broken enough sticks, keep sampling.
    _, _, self._theta, self._beta, samples = tf.while_loop(
        self._sample_n_cond, self._sample_n_body,
        loop_vars=[k, bools, self.theta, self.beta, draws],
        shape_invariants=[
            k.shape, bools.shape, theta_shape, beta_shape, draws.shape])

    return samples

  def _sample_n_cond(self, k, bools, theta, beta, draws):
    # Proceed if at least one bool is True.
    return tf.reduce_any(bools)

  def _sample_n_body(self, k, bools, theta, beta, draws):
    n, batch_shape, event_shape, rank = self._temp_scope

    # If necessary, break a new piece of stick, i.e.
    # add a new persistent atom to theta and sample another beta
    theta, beta = tf.cond(
        tf.shape(theta)[0] - 1 >= k,
        lambda: (theta, beta),
        lambda: (
            tf.concat(
                [theta, tf.expand_dims(self.base.sample(batch_shape), 0)], 0),
            tf.concat(
                [beta, tf.expand_dims(self._betadist.sample(), 0)], 0)))
    theta_k = tf.gather(theta, k)
    beta_k = tf.gather(beta, k)

    # Assign True samples to the new theta_k.
    if len(bools.shape) <= 1:
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

    # Flip coins according to stick probabilities.
    flips = Bernoulli(p=beta_k).sample(n)
    # If coin lands heads, assign sample's corresponding bool to False
    # (this ends its "while loop").
    bools = tf.where(tf.cast(flips, tf.bool), tf.zeros_like(bools), bools)
    return k + 1, bools, theta, beta, draws
