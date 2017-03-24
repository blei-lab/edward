"""A mixture distribution where all components are of the same family."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.util import get_dims, logit
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class ParamMixture(distribution.Distribution):
  def __init__(self,
               cat, components,
               validate_args=False,
               allow_nan_stats=True,
               name="ParamMixture"):
    parameters = locals()
    with tf.name_scope(name, values=[cat, components]) as name:
      self._cat = cat
      self._components = components
    super(ParamMixture, self).__init__(
      dtype=components.dtype,
      validate_args=validate_args,
      allow_nan_stats=allow_nan_stats,
      is_continuous=False,
      is_reparameterized=False,
      parameters=parameters,
#       graph_parents=[self._cat,
#                      self._components],
#     super(ParamMixture, self).__init__(
#       dtype=components.dtype,
#       validate_args=validate_args,
#       allow_nan_stats=allow_nan_stats,
#       reparameterization_type=distribution.NOT_REPARAMETERIZED,
#       parameters=parameters,
#       graph_parents=[self._cat,
#                      self._components],
      name=name)

  @property
  def cat(self):
    return self._cat

  @property
  def components(self):
    return self._components

  def _log_prob(self, x, conjugate=False, **kwargs):
    event_dim = len(self.components.get_event_shape())
    expanded_x = tf.expand_dims(x, -1 - event_dim)
    if conjugate:
      log_probs = self.components.conjugate_log_prob(expanded_x)
    else:
      log_probs = self.components.log_prob(expanded_x)
    selecter = tf.one_hot(self.cat.value(), self.cat.p.get_shape()[-1],
                          dtype=tf.float32)
    return tf.reduce_sum(log_probs * selecter, -1)

  def conjugate_log_prob(self):
    return self._log_prob(self.value(), conjugate=True)

  @distribution.distribution_util.AppendDocstring(
    '''The semantics of `n` may be confusing here. If n is not 1, then
    we will *not* choose new mixture components for each sample. E.g.,
    if `cat` is [0, 1, 3], and `n` is 3, we will sample 3 times from
    component 0, 3 times from component 1, and 3 times from component
    3. We will *not* resample `cat` 3 times.'''
    )
  def _sample_n(self, n, seed=None):
#     if n != 1:
#       raise NotImplementedError('ParamMixture does not allow for'
#                                 ' `sample_shape`s other than 1. Sample size is'
#                                 ' taken from the `cat` input.')
    # TODO(mhoffman): Make this more efficient
    event_dim = len(self.components.get_event_shape())
    new_shape = tf.concat([tf.TensorShape(n), self.cat.get_shape(),
                           self.components.get_shape()], 0)
    all_samples = self.components.sample(new_shape, seed=seed)
    selecter = tf.one_hot(self.cat.value(), self.cat.p.get_shape()[-1],
                          dtype=tf.float32)
    return tf.reduce_sum(all_samples * selecter, -1)

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  @property
  def n(self):
    """Number of samples."""
    return self._n

  def _batch_shape(self):
    return self.cat.get_shape()

  def _event_shape(self):
    return self.components.get_event_shape()

  def _mean(self):
    raise NotImplementedError()

  def _std(self):
    raise NotImplementedError()

  def _variance(self):
    raise NotImplementedError()
