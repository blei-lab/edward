"""The Empirical distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.util import get_dims, logit, tile
from tensorflow.contrib.distributions.python.ops import \
    distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class Empirical(distribution.Distribution):
  """Empirical distribution."""
  def __init__(self,
               params,
               validate_args=False,
               allow_nan_stats=True,
               name="Empirical"):
    with ops.name_scope(name, values=[params]) as ns:
      with ops.control_dependencies([]):
        self._params = array_ops.identity(params, name="params")
        try:
          self._n = get_dims(self._params)[0]
        except:  # scalar params
          self._n = 1

        super(Empirical, self).__init__(
            dtype=self._params.dtype,
            parameters={"params": self._params,
                        "n": self._n},
            is_continuous=False,
            is_reparameterized=True,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"params": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  @property
  def n(self):
    """Number of samples."""
    return self._n

  def _batch_shape(self):
    return array_ops.constant([], dtype=dtypes.int32)

  def _get_batch_shape(self):
    return tensor_shape.scalar()

  def _event_shape(self):
    return ops.convert_to_tensor(self.get_event_shape())

  def _get_event_shape(self):
    return self._params.get_shape()[1:]

  def _mean(self):
    return tf.reduce_mean(self._params, 0)

  def _std(self):
    # broadcasting T x shape - shape = T x shape
    r = self._params - self.mean()
    return tf.sqrt(tf.reduce_mean(tf.square(r), 0))

  def _variance(self):
    return math_ops.square(self.std())

  def _sample_n(self, n, seed=None):
    if self.n != 1:
      logits = logit(tf.ones(self.n, dtype=tf.float32) /
                     tf.cast(self.n, dtype=tf.float32))
      cat = tf.contrib.distributions.Categorical(logits=logits)
      indices = cat._sample_n(n, seed)
      return tf.gather(self._params, indices)
    else:
      multiples = tf.concat(
          [tf.expand_dims(n, 0), [1] * len(self.get_event_shape())], 0)
      return tile(self._params, multiples)
