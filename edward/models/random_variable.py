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
  >>> p = tf.constant(0.5)
  >>> x = Bernoulli(p=p)
  >>>
  >>> z1 = tf.constant([[2.0, 8.0]])
  >>> z2 = tf.constant([[1.0, 2.0]])
  >>> x = Bernoulli(p=tf.matmul(z1, z2))
  >>>
  >>> mu = Normal(mu=tf.constant(0.0), sigma=tf.constant(1.0))
  >>> x = Normal(mu=mu, sigma=tf.constant(1.0))

  Notes
  -----
  ``RandomVariable`` assumes use in a multiple inheritance setting. The
  child class must first inherit ``RandomVariable``, then second inherit a
  class in ``tf.contrib.distributions``. With Python's method resolution
  order, this implies the following during initialization (using
  ``distributions.Bernoulli`` as an example):

  1. Start the ``__init__()`` of the child class, which passes all
     ``*args, **kwargs`` to ``RandomVariable``.
  2. This in turn passes all ``*args, **kwargs`` to
     ``distributions.Bernoulli``, completing the ``__init__()`` of
     ``distributions.Bernoulli``.
  3. Complete the ``__init__()`` of ``RandomVariable``, which calls
    ``self.sample()``, relying on the method from
    ``distributions.Bernoulli``.
  4. Complete the ``__init__()`` of the child class.

  Methods from both ``RandomVariable`` and ``distributions.Bernoulli``
  populate the namespace of the child class. Methods from
  ``RandomVariable`` will take higher priority if there are conflicts.
  """
  def __init__(self, *args, **kwargs):
    # storing args, kwargs for easy graph copying
    self._args = args
    self._kwargs = kwargs

    # need to temporarily pop value before __init__
    value = kwargs.pop('value', None)
    super(RandomVariable, self).__init__(*args, **kwargs)
    if value is not None:
      self._kwargs['value'] = value  # reinsert (needed for copying)

    tf.add_to_collection(RANDOM_VARIABLE_COLLECTION, self)

    if value is not None:
      t_value = tf.convert_to_tensor(value, self.dtype)
      expected_shape = (self.get_batch_shape().as_list() +
                        self.get_event_shape().as_list())
      value_shape = t_value.get_shape().as_list()
      if value_shape != expected_shape:
        raise ValueError(
            "Incompatible shape for initialization argument 'value'. "
            "Expected %s, got %s." % (expected_shape, value_shape))
      else:
        self._value = t_value
    else:
      self._value = self.sample()

  def __str__(self):
    return '<ed.RandomVariable \'' + self.name.__str__() + '\' ' + \
           'shape=' + self._value.get_shape().__str__() + ' ' \
           'dtype=' + self.dtype.__repr__() + \
           '>'

  def __repr__(self):
    return self.__str__()

  def __add__(self, other):
    return tf.add(self, other)

  def __radd__(self, other):
    return tf.add(other, self)

  def __sub__(self, other):
    return tf.subtract(self, other)

  def __rsub__(self, other):
    return tf.subtract(other, self)

  def __mul__(self, other):
    return tf.multiply(self, other)

  def __rmul__(self, other):
    return tf.multiply(other, self)

  def __div__(self, other):
    return tf.div(self, other)

  __truediv__ = __div__

  def __rdiv__(self, other):
    return tf.div(other, self)

  __rtruediv__ = __rdiv__

  def __floordiv__(self, other):
    return tf.floor(tf.div(self, other))

  def __rfloordiv__(self, other):
    return tf.floor(tf.div(other, self))

  def __mod__(self, other):
    return tf.mod(self, other)

  def __rmod__(self, other):
    return tf.mod(other, self)

  def __lt__(self, other):
    return tf.less(self, other)

  def __le__(self, other):
    return tf.less_equal(self, other)

  def __gt__(self, other):
    return tf.greater(self, other)

  def __ge__(self, other):
    return tf.greater_equal(self, other)

  def __and__(self, other):
    return tf.logical_and(self, other)

  def __rand__(self, other):
    return tf.logical_and(other, self)

  def __or__(self, other):
    return tf.logical_or(self, other)

  def __ror__(self, other):
    return tf.logical_or(other, self)

  def __xor__(self, other):
    return tf.logical_xor(self, other)

  def __rxor__(self, other):
    return tf.logical_xor(other, self)

  def __pow__(self, other):
    return tf.pow(self, other)

  def __rpow__(self, other):
    return tf.pow(other, self)

  def __invert__(self):
    return tf.logical_not(self)

  def __neg__(self):
    return tf.negative(self)

  def __abs__(self):
    return tf.abs(self)

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return id(self) == id(other)

  def value(self):
    """Get tensor that the random variable corresponds to."""
    return self._value

  def get_ancestors(self, collection=None):
    """Get ancestor random variables."""
    from edward.util.random_variables import get_ancestors
    return get_ancestors(self, collection)

  def get_children(self, collection=None):
    """Get child random variables."""
    from edward.util.random_variables import get_children
    return get_children(self, collection)

  def get_descendants(self, collection=None):
    """Get descendant random variables."""
    from edward.util.random_variables import get_descendants
    return get_descendants(self, collection)

  def get_parents(self, collection=None):
    """Get parent random variables."""
    from edward.util.random_variables import get_parents
    return get_parents(self, collection)

  def get_siblings(self, collection=None):
    """Get sibling random variables."""
    from edward.util.random_variables import get_siblings
    return get_siblings(self, collection)

  def get_variables(self, collection=None):
    """Get TensorFlow variables that the random variable depends on."""
    from edward.util.random_variables import get_variables
    return get_variables(self, collection)

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
