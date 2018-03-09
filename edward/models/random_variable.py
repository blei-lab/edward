from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from collections import defaultdict
import six

try:
  from tensorflow.python.client.session import \
      register_session_run_conversion_functions
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))

_RANDOM_VARIABLE_COLLECTION = defaultdict(list)


class RandomVariable(object):
  """Base class for random variables.

  A random variable is an object parameterized by tensors. It is
  equipped with methods such as the log-density, mean, and sample.

  It also wraps a tensor, where the tensor corresponds to a sample
  from the random variable. This enables operations on the TensorFlow
  graph, allowing random variables to be used in conjunction with
  other TensorFlow ops.

  The random variable's shape is given by

  `sample_shape + batch_shape + event_shape`,

  where `sample_shape` is an optional argument representing the
  dimensions of samples drawn from the distribution (default is
  a scalar); `batch_shape` is the number of independent random variables
  (determined by the shape of its parameters); and `event_shape` is
  the shape of one draw from the distribution (e.g., `Normal` has a
  scalar `event_shape`; `Dirichlet` has a vector `event_shape`).

  #### Notes

  `RandomVariable` assumes use in a multiple inheritance setting. The
  child class must first inherit `RandomVariable`, then second inherit a
  class in `tf.contrib.distributions`. With Python's method resolution
  order, this implies the following during initialization (using
  `distributions.Bernoulli` as an example):

  1. Start the `__init__()` of the child class, which passes all
     `*args, **kwargs` to `RandomVariable`.
  2. This in turn passes all `*args, **kwargs` to
     `distributions.Bernoulli`, completing the `__init__()` of
     `distributions.Bernoulli`.
  3. Complete the `__init__()` of `RandomVariable`, which calls
     `self.sample()`, relying on the method from
     `distributions.Bernoulli`.
  4. Complete the `__init__()` of the child class.

  Methods from both `RandomVariable` and `distributions.Bernoulli`
  populate the namespace of the child class. Methods from
  `RandomVariable` will take higher priority if there are conflicts.

  #### Examples

  ```python
  p = tf.constant(0.5)
  x = Bernoulli(p)

  z1 = tf.constant([[1.0, -0.8], [0.3, -1.0]])
  z2 = tf.constant([[0.9, 0.2], [2.0, -0.1]])
  x = Bernoulli(logits=tf.matmul(z1, z2))

  mu = Normal(tf.constant(0.0), tf.constant(1.0))
  x = Normal(mu, tf.constant(1.0))
  ```
  """
  def __init__(self, *args, **kwargs):
    """Create a new random variable.

    Args:
      sample_shape: tf.TensorShape.
        Shape of samples to draw from the random variable.
      value: tf.Tensor.
        Fixed tensor to associate with random variable. Must have shape
        `sample_shape + batch_shape + event_shape`.
      collections: list.
        Optional list of graph collections (lists). The random variable is
        added to these collections. Defaults to `[ed.random_variables()]`.
    """
    # pop and store RandomVariable-specific parameters
    sample_shape = kwargs.pop('sample_shape', ())
    value = kwargs.pop('value', None)
    collections = kwargs.pop('collections', ["random_variables"])

    super(RandomVariable, self).__init__(*args, **kwargs)

    self._sample_shape = tf.TensorShape(sample_shape)
    if value is not None:
      t_value = tf.convert_to_tensor(value, self.dtype)
      value_shape = t_value.shape
      expected_shape = self._sample_shape.concatenate(
          self.batch_shape).concatenate(self.event_shape)
      if not value_shape.is_compatible_with(expected_shape):
        raise ValueError(
            "Incompatible shape for initialization argument 'value'. "
            "Expected %s, got %s." % (expected_shape, value_shape))
      else:
        self._value = t_value
    else:
      try:
        self._value = self.sample(self._sample_shape)
      except NotImplementedError:
        raise NotImplementedError(
            "sample is not implemented for {0}. You must either pass in the "
            "value argument or implement sample for {0}."
            .format(self.__class__.__name__))

    for collection in collections:
      if collection == "random_variables":
        collection = _RANDOM_VARIABLE_COLLECTION
      collection[tf.get_default_graph()].append(self)

  @property
  def sample_shape(self):
    """Sample shape of random variable."""
    return self._sample_shape

  @property
  def shape(self):
    """Shape of random variable."""
    return self.value.shape

  @property
  def value(self):
    """Get tensor that the random variable corresponds to."""
    return self._value

  def __str__(self):
    if not hasattr(self.value, "numpy"):
      name = self.name
    else:
      name = numpy_text(self.value)
    return "RandomVariable(\"%s\"%s%s%s)" % (
        name,
        (", shape=%s" % self.shape)
        if self.shape.ndims is not None else "",
        (", dtype=%s" % self.dtype.name) if self.dtype else "",
        (", device=%s" % self.value.device) if self.value.device else "")

  def __repr__(self):
    string = "<ed.RandomVariable '%s' shape=%s dtype=%s>" % (
        self.name, self.shape, self.dtype.name)
    if hasattr(self.value, "numpy"):
      string = string[:-1] + " numpy=%s>" % (
          numpy_text(self.value, is_repr=True))
    return string

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return id(self) == id(other)

  def __iter__(self):
    raise TypeError("'RandomVariable' object is not iterable.")

  def __bool__(self):
    raise TypeError(
        "Using a `ed.RandomVariable` as a Python `bool` is not allowed. "
        "Use `if t is not None:` instead of `if t:` to test if a "
        "random variable is defined, and use TensorFlow ops such as "
        "tf.cond to execute subgraphs conditioned on a draw from "
        "a random variable.")

  def __nonzero__(self):
    raise TypeError(
        "Using a `ed.RandomVariable` as a Python `bool` is not allowed. "
        "Use `if t is not None:` instead of `if t:` to test if a "
        "random variable is defined, and use TensorFlow ops such as "
        "tf.cond to execute subgraphs conditioned on a draw from "
        "a random variable.")

  def eval(self, session=None, feed_dict=None):
    """In a session, computes and returns the value of this random variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.

    Args:
      session: tf.BaseSession.
        The `tf.Session` to use to evaluate this random variable. If
        none, the default session is used.
      feed_dict: dict.
        A dictionary that maps `tf.Tensor` objects to feed values. See
        `tf.Session.run()` for a description of the valid feed values.

    #### Examples

    ```python
    x = Normal(0.0, 1.0)
    with tf.Session() as sess:
      # Usage passing the session explicitly.
      print(x.eval(sess))
      # Usage with the default session.  The 'with' block
      # above makes 'sess' the default session.
      print(x.eval())
    ```
    """
    return self.value.eval(session=session, feed_dict=feed_dict)

  def numpy(self):
    """Value as NumPy array, only available for TF Eager."""
    return self.value.numpy()

  def get_ancestors(self, collection=None):
    """Get ancestor random variables."""
    from edward.models.queries import get_ancestors
    return get_ancestors(self, collection)

  def get_blanket(self, collection=None):
    """Get the random variable's Markov blanket."""
    from edward.models.queries import get_blanket
    return get_blanket(self, collection)

  def get_children(self, collection=None):
    """Get child random variables."""
    from edward.models.queries import get_children
    return get_children(self, collection)

  def get_descendants(self, collection=None):
    """Get descendant random variables."""
    from edward.models.queries import get_descendants
    return get_descendants(self, collection)

  def get_parents(self, collection=None):
    """Get parent random variables."""
    from edward.models.queries import get_parents
    # The backward pass requires TF graph traversal. In general, consider
    # primitive -> black box function (TF ops) -> primitive. To go to parent
    # primitive, we traverse black box function.
    parents = []
    for node in six.itervalues(self.parameters):
      if isinstance(node,
                    (tf.Variable, tf.SparseTensor, tf.Tensor, RandomVariable)):
        parents.extend(get_parents(node))
      if isinstance(node, RandomVariable):
        parents.append(node)
    return parents

  def get_siblings(self, collection=None):
    """Get sibling random variables."""
    from edward.models.queries import get_siblings
    return get_siblings(self, collection)

  def get_variables(self, collection=None):
    """Get TensorFlow variables that the random variable depends on."""
    from edward.models.queries import get_variables
    return get_variables(self, collection)

  def get_shape(self):
    """Get shape of random variable."""
    return self.shape

  @staticmethod
  def _overload_all_operators():
    """Register overloads for all operators."""
    for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
      RandomVariable._overload_operator(operator)

  @staticmethod
  def _overload_operator(operator):
    """Defer an operator overload to `tf.Tensor`.

    We pull the operator out of tf.Tensor dynamically to avoid ordering issues.

    Args:
      operator: string. The operator name.
    """
    def _run_op(a, *args):
      return getattr(tf.Tensor, operator)(a.value, *args)
    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = getattr(tf.Tensor, operator).__doc__
    except AttributeError:
      pass

    setattr(RandomVariable, operator, _run_op)

  # "This enables the Variable's overloaded "right" binary operators to
  # run when the left operand is an ndarray, because it accords the
  # Variable class higher priority than an ndarray, or a numpy matrix."
  # Taken from implementation of tf.Tensor.
  __array_priority__ = 100

  @staticmethod
  def _session_run_conversion_fetch_function(tensor):
    return ([tensor.value], lambda val: val[0])

  @staticmethod
  def _session_run_conversion_feed_function(feed, feed_val):
    return [(feed.value, feed_val)]

  @staticmethod
  def _session_run_conversion_feed_function_for_partial_run(feed):
    return [feed.value]

  @staticmethod
  def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
    _ = name, as_ref
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    return v.value


def numpy_text(tensor, is_repr=False):  # utility fn from TF Eager codebase
  """Human readable representation of a tensor's numpy value."""
  if tensor.dtype.is_numpy_compatible:
    text = repr(tensor.numpy()) if is_repr else str(tensor.numpy())
  else:
    text = "<unprintable>"
  if "\n" in text:
    text = "\n" + text
  return text


def random_variables(graph=None):
  """Return all random variables in the TensorFlow graph.

  Args:
    graph: TensorFlow graph.

  Returns:
    list of RandomVariable.
  """
  if graph is None:
    graph = tf.get_default_graph()

  return _RANDOM_VARIABLE_COLLECTION[graph]


RandomVariable._overload_all_operators()

register_session_run_conversion_functions(
    RandomVariable,
    RandomVariable._session_run_conversion_fetch_function,
    RandomVariable._session_run_conversion_feed_function,
    RandomVariable._session_run_conversion_feed_function_for_partial_run)

tf.register_tensor_conversion_function(
    RandomVariable, RandomVariable._tensor_conversion_function)
