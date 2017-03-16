from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution


class PointMass(RandomVariable, Distribution):
  """PointMass random variable.

  It is analogous to an Empirical random variable with one sample, but
  its parameter argument does not have an outer dimension.
  """
  def __init__(self, params, validate_args=False, allow_nan_stats=True,
               name="PointMass", *args, **kwargs):
    with tf.name_scope(name, values=[params]) as ns:
      with tf.control_dependencies([]):
        self._params = tf.identity(params, name="params")
        super(PointMass, self).__init__(
            dtype=self._params.dtype,
            parameters={"params": self._params},
            is_continuous=False,
            is_reparameterized=True,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns,
            *args, **kwargs)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"params": tf.expand_dims(
        tf.convert_to_tensor(sample_shape, dtype=tf.int32), 0)}

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  def _batch_shape(self):
    return tf.convert_to_tensor(self.get_batch_shape())

  def _get_batch_shape(self):
    return tf.TensorShape([])

  def _event_shape(self):
    return tf.convert_to_tensor(self.get_event_shape())

  def _get_event_shape(self):
    return self._params.get_shape()

  def _mean(self):
    return self._params

  def _std(self):
    return 0.0 * tf.ones_like(self._params)

  def _variance(self):
    return tf.square(self.std())

  def _sample_n(self, n, seed=None):
    input_tensor = self._params
    input_tensor = tf.expand_dims(input_tensor, 0)
    multiples = tf.concat(
        [tf.expand_dims(n, 0), [1] * len(self.get_event_shape())], 0)
    return tf.tile(input_tensor, multiples)
