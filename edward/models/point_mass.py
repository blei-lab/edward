"""The Point Mass distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.util import tile
from tensorflow.contrib.distributions.python.ops import \
    distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class PointMass(distribution.Distribution):
  """PointMass distribution.

  It is analogous to an Empirical random variable with one sample, but
  its parameter argument does not have an outer dimension.
  """
  def __init__(self,
               params,
               validate_args=False,
               allow_nan_stats=True,
               name="PointMass"):
    with ops.name_scope(name, values=[params]) as ns:
      with ops.control_dependencies([]):
        self._params = array_ops.identity(params, name="params")
        super(PointMass, self).__init__(
            dtype=self._params.dtype,
            parameters={"params": self._params},
            is_continuous=False,
            is_reparameterized=True,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"params": tf.expand_dims(
        ops.convert_to_tensor(sample_shape, dtype=dtypes.int32), 0)}

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  def _batch_shape(self):
    return array_ops.constant([], dtype=dtypes.int32)

  def _get_batch_shape(self):
    return tensor_shape.scalar()

  def _event_shape(self):
    return array_ops.shape(self._params)

  def _get_event_shape(self):
    return self._params.get_shape()

  def _mean(self):
    return self._params

  def _std(self):
    return 0.0 * array_ops.ones_like(self._params)

  def _variance(self):
    return math_ops.square(self.std())

  def _sample_n(self, n, seed=None):
    multiples = tf.concat(
        [tf.expand_dims(n, 0), [1] * len(self.get_event_shape())], 0)
    return tile(self._params, multiples)
