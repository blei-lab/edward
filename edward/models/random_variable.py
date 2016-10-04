from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

RANDOM_VARIABLE_COLLECTION = "_random_variable_collection_"


class RandomVariable(object):
  """Base class for random variables.

  A random variable is an object parameterized by tensors. It is
  equipped with methods such as the log-density, mean, and sample.

  It also wraps a tensor, where the tensor corresponds to a sample
  from the random variable. This enables operations on the TensorFlow
  graph, allowing random variables to be used in conjunction with
  other TensorFlow ops.

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
  RandomVariable assumes use in a multiple inheritance setting. The
  child class must first inherit RandomVariable, then second inherit a
  class in tf.contrib.distributions. With Python's method resolution
  order, this implies the following during initialization (using
  distributions.Bernoulli as an example):

  1. Start the __init__() of the child class, which passes all *args,
     **kwargs to RandomVariable.
  2. This in turn passes all *args, **kwargs to
     distributions.Bernoulli, completing the __init__() of
     distributions.Bernoulli.
  3. Complete the __init__() of RandomVariable, which calls
     self.sample(), relying on the method from distributions.Bernoulli.
  4. Complete the __init__() of the child class.

  Methods from both RandomVariable and distributions.Bernoulli
  populate the namespace of the child class. Methods from
  RandomVariable will take higher priority if there are conflicts.
  """
  def __init__(self, *args, **kwargs):
    # storing args, kwargs for easy graph copying
    self._args = args
    self._kwargs = kwargs
    super(RandomVariable, self).__init__(*args, **kwargs)
    tf.add_to_collection(RANDOM_VARIABLE_COLLECTION, self)
    self._value = self.sample()

  def __str__(self):
    return '<ed.RandomVariable \'' + self.name.__str__() + '\' ' + \
           'shape=' + self._value.get_shape().__str__() + ' ' \
           'dtype=' + self.dtype.__repr__() + \
           '>'

  def __repr__(self):
    return self.__str__()

  def value(self):
    return self._value

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
