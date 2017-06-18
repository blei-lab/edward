from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution

try:
  from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class distributions_PointMass(Distribution):
  """PointMass random variable.

  It is analogous to an Empirical random variable with one sample, but
  its parameter argument does not have an outer dimension.
  """
  def __init__(self,
               params,
               validate_args=False,
               allow_nan_stats=True,
               name="PointMass"):
    """Initialize a `PointMass` random variable.

    Args:
      params: tf.Tensor.
        The location with all probability mass.

    #### Examples

    ```python
    # scalar
    x = PointMass(params=28.3)
    assert x.shape == ()

    # 5 x 2 x 3 tensor
    dp = PointMass(params=tf.zeros([5, 2, 3]))
    assert x.shape == (5, 2, 3)
    ```
    """
    parameters = locals()
    with tf.name_scope(name, values=[params]):
      with tf.control_dependencies([]):
        self._params = tf.identity(params, name="params")

    super(distributions_PointMass, self).__init__(
        dtype=self._params.dtype,
        reparameterization_type=FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._params],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"params": tf.expand_dims(
        tf.convert_to_tensor(sample_shape, dtype=tf.int32), 0)}

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  def _batch_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape_tensor(self):
    return tf.shape(self.params)

  def _event_shape(self):
    return self.params.shape

  def _mean(self):
    return self.params

  def _stddev(self):
    return 0.0 * tf.ones_like(self.params)

  def _variance(self):
    return tf.square(self.stddev())

  def _sample_n(self, n, seed=None):
    input_tensor = self.params
    input_tensor = tf.expand_dims(input_tensor, 0)
    multiples = tf.concat(
        [tf.expand_dims(n, 0), [1] * len(self.event_shape)], 0)
    return tf.tile(input_tensor, multiples)


# Generate random variable class similar to autogenerated ones from TensorFlow.
_name = 'PointMass'
_candidate = distributions_PointMass
_globals = globals()
params = {'__doc__': _candidate.__doc__}
_globals[_name] = type(_name, (RandomVariable, _candidate), params)
