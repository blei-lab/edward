# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The Empirical distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import categorical


class Empirical(distribution.Distribution):
  """Empirical random variable.

  #### Examples

  ```python
  # 100 samples of a scalar
  x = Empirical(loc=tf.zeros(100))
  assert x.shape == ()

  # 5 samples of a 2 x 3 matrix
  dp = Empirical(loc=tf.zeros([5, 2, 3]))
  assert x.shape == (2, 3)
  ```
  """
  def __init__(self,
               loc,
               validate_args=False,
               allow_nan_stats=True,
               name="Empirical"):
    """Initialize an `Empirical` random variable.

    Args:
      loc: tf.Tensor.
      Collection of samples. Its outer (left-most) dimension
      determines the number of samples.
    """
    parameters = locals()
    with ops.name_scope(name, values=[loc]):
      with ops.control_dependencies([]):
        self._loc = array_ops.identity(loc, name="loc")
        try:
          self._n = array_ops.shape(self._loc)[0]
        except ValueError:  # scalar loc
          self._n = constant_op.constant(1)

    super(distributions_Empirical, self).__init__(
        dtype=self._loc.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._n],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"loc": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def loc(self):
    """Distribution parameter."""
    return self._loc

  @property
  def n(self):
    """Number of samples."""
    return self._n

  def _batch_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _batch_shape(self):
    return tensor_shape.scalar()

  def _event_shape_tensor(self):
    return array_ops.shape(self.loc)[1:]

  def _event_shape(self):
    return self.loc.shape[1:]

  def _mean(self):
    return math_ops.reduce_mean(self.loc, 0)

  def _stddev(self):
    # broadcasting n x shape - shape = n x shape
    r = self.loc - self.mean()
    return math_ops.sqrt(math_ops.reduce_mean(math_ops.square(r), 0))

  def _variance(self):
    return math_ops.square(self.stddev())

  def _sample_n(self, n, seed=None):
    input_tensor = self.loc
    if len(input_tensor.shape) == 0:
      input_tensor = array_ops.expand_dims(input_tensor, 0)
      multiples = array_ops.concat(
          [array_ops.expand_dims(n, 0), [1] * len(self.event_shape)], 0)
      return array_ops.tile(input_tensor, multiples)
    else:
      probs = array_ops.ones([self.n]) / math_ops.cast(self.n, dtype=dtypes.float32)
      cat = categorical.Categorical(probs)
      indices = cat._sample_n(n, seed)
      tensor = array_ops.gather(input_tensor, indices)
      return tensor
