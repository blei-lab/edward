from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution

try:
  from edward.models.random_variables import Bernoulli, Beta
  from tensorflow.contrib.distributions import NOT_REPARAMETERIZED
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class distributions_DirichletProcess(Distribution):
  """Dirichlet process :math:`\mathcal{DP}(\\alpha, H)`.

  It has two parameters: a positive real value :math:`\\alpha`, known
  as the concentration parameter (``concentration``), and a base
  distribution :math:`H` (``base``).
  """
  def __init__(self,
               concentration,
               base,
               validate_args=False,
               allow_nan_stats=True,
               name="DirichletProcess"):
    """Initialize a batch of Dirichlet processes.

    Parameters
    ----------
    concentration : tf.Tensor
      Concentration parameter. Must be positive real-valued. Its shape
      determines the number of independent DPs (batch shape).
    base : RandomVariable
      Base distribution. Its shape determines the shape of an
      individual DP (event shape).

    Examples
    --------
    >>> # scalar concentration parameter, scalar base distribution
    >>> dp = DirichletProcess(0.1, Normal(loc=0.0, scale=1.0))
    >>> assert dp.shape == ()
    >>>
    >>> # vector of concentration parameters, matrix of Exponentials
    >>> dp = DirichletProcess(tf.constant([0.1, 0.4]),
    ...                       Exponential(lam=tf.ones([5, 3])))
    >>> assert dp.shape == (2, 5, 3)
    """
    parameters = locals()
    with tf.name_scope(name, values=[concentration]):
      with tf.control_dependencies([
          tf.assert_positive(concentration),
      ] if validate_args else []):
        if validate_args and isinstance(base, RandomVariable):
          raise TypeError("base must be a ed.RandomVariable object.")

        self._concentration = tf.identity(concentration, name="concentration")
        self._base = base

        # Form empty tensor to store atom locations.
        self._locs = tf.zeros(
            [0] + self.batch_shape.as_list() + self.event_shape.as_list(),
            dtype=self._base.dtype)

        # Instantiate distribution to draw mixing proportions.
        self._probs_dist = Beta(tf.ones_like(self._concentration),
                                self._concentration,
                                collections=[])
        # Form empty tensor to store mixing proportions.
        self._probs = tf.zeros(
            [0] + self.batch_shape.as_list(),
            dtype=self._probs_dist.dtype)

    super(distributions_DirichletProcess, self).__init__(
        dtype=tf.int32,
        reparameterization_type=NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._concentration, self._locs, self._probs],
        name=name)

  @property
  def base(self):
    """Base distribution used for drawing the atom locations."""
    return self._base

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def locs(self):
    """Atom locations. It has shape [None] + batch_shape +
    event_shape, where the first dimension is the number of atoms,
    instantiated only as needed."""
    return self._locs

  @property
  def probs(self):
    """Mixing proportions. It has shape [None] + batch_shape, where
    the first dimension is the number of atoms, instantiated only as
    needed."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.shape(self.concentration)

  def _batch_shape(self):
    return self.concentration.shape

  def _event_shape_tensor(self):
    return tf.shape(self.base)

  def _event_shape(self):
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

    batch_shape = self.batch_shape.as_list()
    event_shape = self.event_shape.as_list()
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

    # Calculate shape invariance conditions for locs and probs as these
    # can change shape between loop iterations.
    locs_shape = tf.TensorShape([None])
    probs_shape = tf.TensorShape([None])
    if len(self.locs.shape) > 1:
      locs_shape = locs_shape.concatenate(self.locs.shape[1:])
      probs_shape = probs_shape.concatenate(self.probs.shape[1:])

    # While we have not broken enough sticks, keep sampling.
    _, _, self._locs, self._probs, samples = tf.while_loop(
        self._sample_n_cond, self._sample_n_body,
        loop_vars=[k, bools, self.locs, self.probs, draws],
        shape_invariants=[
            k.shape, bools.shape, locs_shape, probs_shape, draws.shape])

    return samples

  def _sample_n_cond(self, k, bools, locs, probs, draws):
    # Proceed if at least one bool is True.
    return tf.reduce_any(bools)

  def _sample_n_body(self, k, bools, locs, probs, draws):
    n, batch_shape, event_shape, rank = self._temp_scope

    # If necessary, break a new piece of stick, i.e.
    # add a new persistent atom location and weight.
    locs, probs = tf.cond(
        tf.shape(locs)[0] - 1 >= k,
        lambda: (locs, probs),
        lambda: (
            tf.concat(
                [locs, tf.expand_dims(self.base.sample(batch_shape), 0)], 0),
            tf.concat(
                [probs, tf.expand_dims(self._probs_dist.sample(), 0)], 0)))
    locs_k = tf.gather(locs, k)
    probs_k = tf.gather(probs, k)

    # Assign True samples to the new locs_k.
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

    locs_k_tile = tf.tile(tf.expand_dims(locs_k, 0), [n] + [1] * (rank - 1))
    draws = tf.where(bools_tile, locs_k_tile, draws)

    # Flip coins according to stick probabilities.
    flips = Bernoulli(probs=probs_k).sample(n)
    # If coin lands heads, assign sample's corresponding bool to False
    # (this ends its "while loop").
    bools = tf.where(tf.cast(flips, tf.bool), tf.zeros_like(bools), bools)
    return k + 1, bools, locs, probs, draws


# Generate random variable class similar to autogenerated ones from TensorFlow.
_name = 'DirichletProcess'
_candidate = distributions_DirichletProcess
_globals = globals()
params = {'__doc__': _candidate.__doc__}
_globals[_name] = type(_name, (RandomVariable, _candidate), params)
