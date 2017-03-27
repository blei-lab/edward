from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution

try:
  from edward.models.random_variables import Categorical
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class ParamMixture(RandomVariable, Distribution):
  """A mixture distribution where all components are of the same family."""
  def __init__(self,
               mixing_weights,
               component_params,
               component_dist,
               validate_args=False,
               allow_nan_stats=True,
               name="ParamMixture",
               *args, **kwargs):
    """Initialize a batch of mixture random variables. The number of
    components is the outer (right-most) dimension of the mixing
    weights, or equivalently, of the components' batch shape.

    Parameters
    ----------
    mixing_weights : tf.Tensor
      (Normalized) weights whose outer (right-most) dimension matches
      the number of components.
    component_params : dict
      Parameters of the per-component distributions.
    component_dist : RandomVariable
      Distribution of each component.

    Examples
    --------
    >>> probs = tf.ones(5) / 5.0
    >>> params = {'mu': tf.zeros(5), 'sigma': tf.ones(5)}
    >>> x = ParamMixture(probs, params, Normal)
    >>> assert x.shape == ()
    >>>
    >>> probs = tf.fill([2, 5], 1.0 / 5.0)
    >>> params = {'p': tf.fill([2, 5], 0.8)}
    >>> x = ParamMixture(probs, params, Bernoulli)
    >>> assert x.shape == (2,)
    """
    parameters = locals()
    parameters.pop("self")
    values = [mixing_weights] + list(six.itervalues(component_params))
    with tf.name_scope(name, values=values) as ns:
      if validate_args:
        if not isinstance(component_params, dict):
          raise TypeError("component_params must be a dict.")
        elif not isinstance(component_dist, RandomVariable):
          raise TypeError("component_dist must be a ed.RandomVariable object.")

      sample_shape = kwargs.get('sample_shape', ())
      self._mixing_weights = mixing_weights
      self._cat = Categorical(p=self._mixing_weights,
                              validate_args=validate_args,
                              allow_nan_stats=allow_nan_stats,
                              sample_shape=sample_shape)
      self._component_params = component_params
      self._components = component_dist(validate_args=validate_args,
                                        allow_nan_stats=allow_nan_stats,
                                        sample_shape=sample_shape,
                                        **component_params)

      with tf.name_scope('means'):
        comp_means = self._components.mean()
        comp_vars = self._components.variance()
        comp_mean_sq = tf.square(comp_means) + comp_vars

        expanded_mix = self._cat.p
        comp_dim = len(self._components.shape)
        if comp_dim >= 2:
          expanded_mix = tf.reshape(
              expanded_mix, self._cat.p.shape.as_list() + [1] * (comp_dim - 2))

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
          is_continuous=self._components.is_continuous,
          is_reparameterized=False,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=ns,
          *args, **kwargs)

  @property
  def cat(self):
    return self._cat

  @property
  def components(self):
    return self._components

  def _batch_shape(self):
    return self.cat.shape

  def _event_shape(self):
    # TODO(mhoffman): This could break if there is only one component.
    return self.components[0].get_event_shape()

  def _log_prob(self, x, conjugate=False, **kwargs):
    event_dim = len(self.components.get_event_shape())
    expanded_x = tf.expand_dims(x, -1 - event_dim)
    if conjugate:
      log_probs = self.components.conjugate_log_prob(expanded_x)
    else:
      log_probs = self.components.log_prob(expanded_x)

    selecter = tf.one_hot(self.cat, self.cat.p.shape[-1],
                          dtype=tf.float32)
    return tf.reduce_sum(log_probs * selecter, -1)

  def conjugate_log_prob(self):
    return self._log_prob(self, conjugate=True)

  def _sample_n(self, n, seed=None):
    # TODO(mhoffman): Make this more efficient
    K = self._mixing_weights.shape[-1]
    selecter = tf.one_hot(self.cat, K, dtype=tf.float32)
    comp_dim = len(self.components.shape)
    if comp_dim > 2:
      selecter = tf.reshape(
          selecter, selecter.shape.as_list() + [1] * (comp_dim - 2))

    return tf.reduce_sum(self.components * selecter, 1)

  def _mean(self):
    return self._mean_val

  def _std(self):
    return self._stddev_val

  def _variance(self):
    return self._variance_val
