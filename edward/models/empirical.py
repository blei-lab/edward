from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution


class Empirical(RandomVariable, Distribution):
  """Empirical random variable."""
  def __init__(self, params, validate_args=False, allow_nan_stats=True,
               name="Empirical", *args, **kwargs):
    """Initialize an ``Empirical`` random variable.

    Parameters
    ----------
    params : tf.Tensor
      Collection of samples. Its outer (left-most) dimension
      determines the number of samples.

    Examples
    --------
    >>> # 100 samples of a scalar
    >>> x = Empirical(params=tf.zeros(100))
    >>> assert x.shape == ()
    >>>
    >>> # 5 samples of a 2 x 3 matrix
    >>> dp = Empirical(params=tf.zeros([5, 2, 3]))
    >>> assert x.shape == (2, 3)
    """
    parameters = locals()
    parameters.pop("self")
    with tf.name_scope(name, values=[params]) as ns:
      with tf.control_dependencies([]):
        self._params = tf.identity(params, name="params")
        try:
          self._n = tf.shape(self._params)[0]
        except ValueError:  # scalar params
          self._n = tf.constant(1)

      super(Empirical, self).__init__(
          dtype=self._params.dtype,
          is_continuous=False,
          is_reparameterized=True,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._params, self._n],
          name=ns,
          *args, **kwargs)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"params": tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  @property
  def n(self):
    """Number of samples."""
    return self._n

  def _batch_shape(self):
    return tf.constant([], dtype=tf.int32)

  def _get_batch_shape(self):
    return tf.TensorShape([])

  def _event_shape(self):
    return tf.shape(self.params)[1:]

  def _get_event_shape(self):
    return self.params.shape[1:]

  def _mean(self):
    return tf.reduce_mean(self.params, 0)

  def _std(self):
    # broadcasting n x shape - shape = n x shape
    r = self.params - self.mean()
    return tf.sqrt(tf.reduce_mean(tf.square(r), 0))

  def _variance(self):
    return tf.square(self.std())

  def _sample_n(self, n, seed=None):
    input_tensor = self.params
    if len(input_tensor.shape) == 0:
      input_tensor = tf.expand_dims(input_tensor, 0)
      multiples = tf.concat(
          [tf.expand_dims(n, 0), [1] * len(self.get_event_shape())], 0)
      return tf.tile(input_tensor, multiples)
    else:
      p = tf.ones([self.n]) / tf.cast(self.n, dtype=tf.float32)
      cat = tf.contrib.distributions.Categorical(p=p)
      indices = cat._sample_n(n, seed)
      tensor = tf.gather(input_tensor, indices)
      return tensor
