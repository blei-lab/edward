"""A mixture distribution where all components are of the same family."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models.random_variables import Categorical
from tensorflow.contrib.distributions.python.ops import distribution


class ParamMixture(RandomVariable, distribution.Distribution):
  def __init__(self,
               mixing_weights, component_params, component_dist,
               sample_shape=(),
               validate_args=False,
               allow_nan_stats=True,
               name="ParamMixture"):
    parameters = locals()
    values = [mixing_weights] + [i for i in component_params.values()]
    with tf.name_scope(name, values=values) as name:
      self._mixing_weights = mixing_weights
      self._component_params = component_params
      self._cat = Categorical(p=self._mixing_weights,
                              sample_shape=sample_shape)
      self._components = component_dist(sample_shape=sample_shape,
                                        **component_params)
    super(ParamMixture, self).__init__(
        dtype=self._components.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        is_continuous=False,
        is_reparameterized=False,
        parameters=parameters,
        name=name,
        sample_shape=sample_shape)

  @property
  def cat(self):
    return self._cat

  def _log_prob(self, x, conjugate=False, **kwargs):
    event_dim = len(self._components.get_event_shape())
    expanded_x = tf.expand_dims(x, -1 - event_dim)
    result = 0
    if conjugate:
      log_probs = self._components.conjugate_log_prob(expanded_x)
    else:
      log_probs = self._components.log_prob(expanded_x)
    selecter = tf.one_hot(self.cat.value(), self.cat.p.get_shape()[-1],
                          dtype=tf.float32)
    result += tf.reduce_sum(log_probs * selecter, -1)
    return result

  def conjugate_log_prob(self):
    return self._log_prob(self.value(), conjugate=True)

  def _sample_n(self, n, seed=None):
    # TODO(mhoffman): Make this more efficient
    selecter = tf.one_hot(self.cat.value(), self.cat.p.get_shape()[-1],
                          dtype=tf.float32)
    return tf.reduce_sum(self._components.value() * selecter, -1)

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
    # TODO(mhoffman): This could break if there is only one component.
    return self._components[0].get_event_shape()

  def _mean(self):
    raise NotImplementedError()

  def _std(self):
    raise NotImplementedError()

  def _variance(self):
    raise NotImplementedError()
