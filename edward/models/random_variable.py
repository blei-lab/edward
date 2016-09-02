from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

RANDOM_VARIABLE_COLLECTION = "_random_variable_collection_"


class RandomVariable(object):
  """
  A random variable is a light wrapper around a tensor. The tensor
  corresponds to samples from the random variable, and the wrapping
  carries properties of the random variable such as its density, mean,
  variance, and sampling.

  Examples
  --------
  >>> p = tf.constant([0.5])
  >>> x = Bernoulli(p=p)
  >>>
  >>> z1 = tf.constant([[2.0, 8.0]])
  >>> z2 = tf.constant([[1.0, 2.0]])
  >>> x = Bernoulli(p=tf.matmul(z1, z2))
  >>>
  >>> mu = Normal(mu=tf.constant(0.0), sigma=tf.constant(1.0)])
  >>> x = Normal(mu=mu, sigma=tf.constant([1.0]))

  Notes
  -----
  This is a simplified version of StochasticTensor in BayesFlow.
  The value type is fixed to SampleAndReshapeValue(), several methods are
  removed, and the distribution's methods populate the namespace for
  class methods.
  """
  def __init__(self, dist_cls, name=None, **dist_args):
    tf.add_to_collection(RANDOM_VARIABLE_COLLECTION, self)
    self._dist_cls = dist_cls
    self._dist_args = dist_args
    with tf.op_scope(dist_args.values(), name, "RandomVariable") as scope:
      self._name = scope
      self._dist = dist_cls(**dist_args)
      self._value = self._dist.sample()

  @property
  def distribution(self):
    return self._dist

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self.distribution.dtype

  @property
  def parameters(self):
    return self.distribution.parameters

  @property
  def is_continuous(self):
    return self.distribution.is_continuous

  @property
  def is_reparameterized(self):
    return self.distribution.is_reparameterized

  @property
  def allow_nan_stats(self):
    return self.distribution.allow_nan_stats

  @property
  def validate_args(self):
    return self.distribution.validate_args

  def value(self):
    return self._value

  def batch_shape(self, *args, **kwargs):
    return self.distribution.batch_shape(*args, **kwargs)

  def get_batch_shape(self, *args, **kwargs):
    return self.distribution.get_batch_shape(*args, **kwargs)

  def event_shape(self, *args, **kwargs):
    return self.distribution.event_shape(*args, **kwargs)

  def get_event_shape(self, *args, **kwargs):
    return self.distribution.get_event_shape(*args, **kwargs)

  def sample(self, *args, **kwargs):
    return self.distribution.sample(*args, **kwargs)

  def sample_n(self, *args, **kwargs):
    return self.distribution.sample_n(*args, **kwargs)

  def log_prob(self, *args, **kwargs):
    return self.distribution.log_prob(*args, **kwargs)

  def prob(self, *args, **kwargs):
    return self.distribution.prob(*args, **kwargs)

  def log_cdf(self, *args, **kwargs):
    return self.distribution.log_cdf(*args, **kwargs)

  def cdf(self, *args, **kwargs):
    return self.distribution.cdf(*args, **kwargs)

  def entropy(self, *args, **kwargs):
    return self.distribution.entropy(*args, **kwargs)

  def mean(self, *args, **kwargs):
    return self.distribution.mean(*args, **kwargs)

  def variance(self, *args, **kwargs):
    return self.distribution.variance(*args, **kwargs)

  def std(self, *args, **kwargs):
    return self.distribution.std(*args, **kwargs)

  def mode(self, *args, **kwargs):
    return self.distribution.mode(*args, **kwargs)

  def log_pdf(self, *args, **kwargs):
    return self.distribution.log_pdf(*args, **kwargs)

  def pdf(self, *args, **kwargs):
    return self.distribution.pdf(*args, **kwargs)

  def log_pmf(self, *args, **kwargs):
    return self.distribution.log_pmf(*args, **kwargs)

  def pmf(self, *args, **kwargs):
    return self.distribution.pmf(*args, **kwargs)

  def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      raise ValueError("%s: Ref type is not supported." % v)
    return v.value()


tf.register_tensor_conversion_function(
    RandomVariable, RandomVariable._tensor_conversion_function)
