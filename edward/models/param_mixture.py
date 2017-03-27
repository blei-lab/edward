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
    values = [mixing_weights] + list(component_params.values())
    with tf.name_scope(name, values=values) as name:
      self._mixing_weights = mixing_weights
      self._component_params = component_params
      self._cat = Categorical(p=self._mixing_weights,
                              sample_shape=sample_shape)
      self._components = component_dist(sample_shape=sample_shape,
                                        **component_params)

      with tf.name_scope('means'):
        comp_means = self._components.mean()
        comp_vars = self._components.variance()
        comp_mean_sq = tf.square(comp_means) + comp_vars
        expanded_mix = self._cat.p
        comp_dim = self._comp_dim()
        if comp_dim > 1:
          new_shape = tf.concat([self._cat.p.get_shape(),
                                 tf.ones(comp_dim - 2, dtype=tf.int32)], 0)
          expanded_mix = tf.reshape(expanded_mix, new_shape)
        self._mean_val = tf.reduce_sum(comp_means * expanded_mix, 0,
                                       name='mean')
        mean_sq_val = tf.reduce_sum(comp_mean_sq * expanded_mix, 0,
                                    name='mean_squared')
        self._variance_val = tf.subtract(mean_sq_val,
                                         tf.square(self._mean_val),
                                         name='variance')
        self._stddev_val = tf.sqrt(self._variance_val, name='stddev')
      
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

  @property
  def components(self):
    return self._components

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
    K = self.cat.p.get_shape()[-1]
    selecter = tf.one_hot(self.cat.value(), K, dtype=tf.float32)
    comp_dim = self._comp_dim()
    if comp_dim > 2:
      new_shape = tf.concat([selecter.get_shape(),
                             tf.ones(comp_dim - 2, dtype=tf.int32)], 0)
      selecter = tf.reshape(selecter, new_shape)
                            
    return tf.reduce_sum(self._components.value() * selecter, 1)

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  @property
  def n(self):
    """Number of samples."""
    return self._n

  def _comp_dim(self):
    return len(self._components.value().get_shape())

  def _batch_shape(self):
    return self.cat.get_shape()

  def _event_shape(self):
    # TODO(mhoffman): This could break if there is only one component.
    return self._components[0].get_event_shape()

  def _mean(self):
    return self._mean_val

  def _std(self):
    return self._stddev_val

  def _variance(self):
    return self._variance_val
