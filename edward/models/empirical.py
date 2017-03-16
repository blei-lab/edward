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
    with tf.name_scope(name, values=[params]) as ns:
      with tf.control_dependencies([]):
        self._params = tf.identity(params, name="params")
        try:
          self._n = self._params.get_shape().as_list()[0]
        except IndexError:  # scalar params
          self._n = 1

        super(Empirical, self).__init__(
            dtype=self._params.dtype,
            parameters={"params": self._params,
                        "n": self._n},
            is_continuous=False,
            is_reparameterized=True,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
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
    return tf.convert_to_tensor(self.get_batch_shape())

  def _get_batch_shape(self):
    return tf.TensorShape([])

  def _event_shape(self):
    return tf.convert_to_tensor(self.get_event_shape())

  def _get_event_shape(self):
    return self._params.get_shape()[1:]

  def _mean(self):
    return tf.reduce_mean(self._params, 0)

  def _std(self):
    # broadcasting T x shape - shape = T x shape
    r = self._params - self.mean()
    return tf.sqrt(tf.reduce_mean(tf.square(r), 0))

  def _variance(self):
    return tf.square(self.std())

  def _sample_n(self, n, seed=None):
    input_tensor = self._params
    if len(input_tensor.get_shape()) == 0:
      input_tensor = tf.expand_dims(input_tensor, 0)
      multiples = tf.concat(
          [tf.expand_dims(n, 0), [1] * len(self.get_event_shape())], 0)
      return tf.tile(input_tensor, multiples)
    else:
      p = tf.ones(self.n, dtype=tf.float32) / self.n
      cat = tf.contrib.distributions.Categorical(p=p)
      indices = cat._sample_n(n, seed)
      tensor = tf.gather(input_tensor, indices)
      return tensor
