from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from copy import deepcopy
from edward.models.random_variable import RandomVariable
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework.ops import set_shapes_for_outputs
from tensorflow.python.util import compat

distributions = tf.contrib.distributions


def copy(org_instance, dict_swap=None, scope="copied",
         replace_itself=False, copy_q=False):
  """Build a new node in the TensorFlow graph from `org_instance`,
  where any of its ancestors existing in `dict_swap` are
  replaced with `dict_swap`'s corresponding value.

  The copying is done recursively, so any `Operation` whose output
  is required to evaluate `org_instance` is also copied (if it isn't
  already copied within the new scope). This is with the exception of
  `tf.Variable`s and `tf.placeholder`s, which are reused and not newly copied.

  Parameters
  ----------
  org_instance : RandomVariable, tf.Variable, tf.Tensor, or tf.Operation
    Node to add in graph with replaced ancestors.
  dict_swap : dict, optional
    Random variables, variables, tensors, or operations to swap with.
    Its keys are what `org_instance` may depend on, and its values are
    the corresponding object (not necessarily of the same class
    instance, but must have the same type, e.g., float32) that is used
    in exchange.
  scope : str, optional
    A scope for the new node(s). This is used to avoid name
    conflicts with the original node(s).
  replace_itself : bool, optional
    Whether to replace `org_instance` itself if it exists in
    `dict_swap`. (This is used for the recursion.)
  copy_q : bool, optional
    Whether to copy the replaced tensors too (if not already
    copied within the new scope). Otherwise will reuse them.

  Returns
  -------
  RandomVariable, tf.Variable, tf.Tensor, or tf.Operation
    The copied node.

  Raises
  ------
  TypeError
    If `org_instance` is not one of the above types.

  Examples
  --------
  >>> x = tf.constant(2.0)
  >>> y = tf.constant(3.0)
  >>> z = x * y
  >>>
  >>> qx = tf.constant(4.0)
  >>> # The TensorFlow graph is currently
  >>> # `x` -> `z` <- y`, `qx`
  >>>
  >>> # This adds a subgraph with newly copied nodes,
  >>> # `copied/qx` -> `copied/z` <- `copied/y`
  >>> z_new = copy(z, {x: qx})
  >>>
  >>> sess = tf.Session()
  >>> sess.run(z)
  6.0
  >>> sess.run(z_new)
  12.0
  """
  if not isinstance(org_instance, RandomVariable) and \
     not isinstance(org_instance, tf.Variable) and \
     not isinstance(org_instance, tf.Tensor) and \
     not isinstance(org_instance, tf.Operation):
    raise TypeError("Could not copy instance: " + str(org_instance))

  if dict_swap is None:
    dict_swap = {}

  # Swap instance if in dictionary.
  if org_instance in dict_swap and replace_itself:
    org_instance = dict_swap[org_instance]
    if not copy_q:
      return org_instance
  elif isinstance(org_instance, tf.Tensor) and replace_itself:
    # Deal with case when `org_instance` is the associated tensor
    # from the RandomVariable, e.g., `z.value()`. If
    # `dict_swap={z: qz}`, we aim to swap it with `qz.value()`.
    for key, value in six.iteritems(dict_swap):
      if isinstance(key, RandomVariable):
        if org_instance == key.value():
          if isinstance(value, RandomVariable):
            org_instance = value.value()
          else:
            org_instance = value

          if not copy_q:
            return org_instance
          break

  graph = tf.get_default_graph()
  new_name = scope + '/' + org_instance.name

  # If an instance of the same name exists, return appropriately.
  # Do this for random variables.
  random_variables = {x.name: x for x in
                      graph.get_collection('_random_variable_collection_')}
  if new_name in random_variables:
    return random_variables[new_name]

  # Do this for tensors and operations.
  try:
    already_present = graph.as_graph_element(new_name,
                                             allow_tensor=True,
                                             allow_operation=True)
    return already_present
  except:
    pass

  # If instance is a variable, return it; do not re-copy any.
  # Note we check variables via their name and not their type. This
  # is because if we get variables through an op's inputs, it has
  # type tf.Tensor: we can only tell it is a variable via its name.
  variables = {x.name: x for x in graph.get_collection(tf.GraphKeys.VARIABLES)}
  if org_instance.name in variables:
    return graph.get_tensor_by_name(variables[org_instance.name].name)

  # Do the same for placeholders. Same logic holds.
  # Note this assumes that placeholders are all in this collection.
  placeholders = {x.name: x for x in graph.get_collection('PLACEHOLDERS')}
  if org_instance.name in placeholders:
    return graph.get_tensor_by_name(placeholders[org_instance.name].name)

  if isinstance(org_instance, RandomVariable):
    rv = org_instance

    # If it has copiable arguments, copy them.
    args = []
    for arg in rv._args:
      if isinstance(arg, RandomVariable) or \
         isinstance(arg, tf.Variable) or \
         isinstance(arg, tf.Tensor) or \
         isinstance(arg, tf.Operation):
         arg = copy(arg, dict_swap, scope, True, copy_q)

      args.append(arg)

    kwargs = {}
    for key, value in six.iteritems(rv._kwargs):
      if isinstance(value, RandomVariable) or \
         isinstance(value, tf.Variable) or \
         isinstance(value, tf.Tensor) or \
         isinstance(value, tf.Operation):
         value = copy(value, dict_swap, scope, True, copy_q)

      kwargs[key] = value

    kwargs['name'] = new_name
    # Create new random variable with copied arguments.
    new_rv = rv.__class__(*args, **kwargs)
    return new_rv
  elif isinstance(org_instance, tf.Tensor):
    tensor = org_instance

    # A tensor is one of the outputs of its underlying
    # op. Therefore copy the op itself.
    op = tensor.op
    new_op = copy(op, dict_swap, scope, True, copy_q)

    output_index = op.outputs.index(tensor)
    new_tensor = new_op.outputs[output_index]

    # Add copied tensor to collections that the original one is in.
    for name, collection in tensor.graph._collections.items():
      if tensor in collection:
        graph.add_to_collection(name, new_tensor)

    return new_tensor
  else:  # tf.Operation
    op = org_instance

    # If it has an original op, copy it.
    if op._original_op is not None:
      new_original_op = copy(op._original_op, dict_swap, scope, True, copy_q)
    else:
      new_original_op = None

    # If it has control inputs, copy them.
    new_control_inputs = []
    for x in op.control_inputs:
      elem = copy(x, dict_swap, scope, True, copy_q)
      if not isinstance(elem, tf.Operation):
        elem = tf.convert_to_tensor(elem)

      new_control_inputs += [elem]

    # If it has inputs, copy them.
    new_inputs = []
    for x in op.inputs:
      elem = copy(x, dict_swap, scope, True, copy_q)
      if not isinstance(elem, tf.Operation):
        elem = tf.convert_to_tensor(elem)

      new_inputs += [elem]

    # Make a copy of the node def.
    # As an instance of tensorflow.core.framework.graph_pb2.NodeDef, it
    # stores string-based info such as name, device, and type of the op.
    # It is unique to every Operation instance.
    new_node_def = deepcopy(op.node_def)
    new_node_def.name = new_name

    # Copy the other inputs needed for initialization.
    output_types = op._output_types[:]
    input_types = op._input_types[:]

    # Make a copy of the op def.
    # It is unique to every Operation type.
    op_def = deepcopy(op.op_def)

    ret = tf.Operation(new_node_def,
                       graph,
                       new_inputs,
                       output_types,
                       new_control_inputs,
                       input_types,
                       new_original_op,
                       op_def)

    # Use Graph's private methods to add the op, following
    # implementation of `tf.Graph().create_op()`.
    compute_shapes = True
    compute_device = True
    op_type = new_name

    if compute_shapes:
      set_shapes_for_outputs(ret)
    graph._add_op(ret)
    graph._record_op_seen_by_control_dependencies(ret)

    if compute_device:
      graph._apply_device_functions(ret)

    if graph._colocation_stack:
      all_colocation_groups = []
      for colocation_op in graph._colocation_stack:
        all_colocation_groups.extend(colocation_op.colocation_groups())
        if colocation_op.device:
          # Make this device match the device of the colocated op, to
          # provide consistency between the device and the colocation
          # property.
          if ret.device and ret.device != colocation_op.device:
            logging.warning("Tried to colocate %s with an op %s that had "
                            "a different device: %s vs %s. "
                            "Ignoring colocation property.",
                            name, colocation_op.name, ret.device,
                            colocation_op.device)
          else:
            ret._set_device(colocation_op.device)

      all_colocation_groups = sorted(set(all_colocation_groups))
      ret.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))

    # Sets "container" attribute if
    # (1) graph._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if (graph._container and
        op_type in graph._registered_ops and
        graph._registered_ops[op_type].is_stateful and
        "container" in ret.node_def.attr and
            not ret.node_def.attr["container"].s):
      ret.node_def.attr["container"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(graph._container)))

    return ret


def cumprod(xs):
  """Cumulative product of a tensor along its outer dimension.

  https://github.com/tensorflow/tensorflow/issues/813

  Parameters
  ----------
  xs : tf.Tensor
    A 1-D or higher tensor.

  Returns
  -------
  tf.Tensor
    A tensor with `cumprod` applied along its outer dimension.

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.
  """
  xs = tf.convert_to_tensor(xs)
  dependencies = [tf.verify_tensor_all_finite(xs, msg='')]
  xs = control_flow_ops.with_dependencies(dependencies, xs)

  values = tf.unpack(xs)
  out = []
  prev = tf.ones_like(values[0])
  for val in values:
    s = prev * val
    out.append(s)
    prev = s

  result = tf.pack(out)
  return result


def dot(x, y):
  """Compute dot product between a 2-D tensor and a 1-D tensor.

  If x is a ``[M x N]`` matrix, then y is a ``M``-vector.

  If x is a ``M``-vector, then y is a ``[M x N]`` matrix.

  Parameters
  ----------
  x : tf.Tensor
    A 1-D or 2-D tensor (see above).
  y : tf.Tensor
    A 1-D or 2-D tensor (see above).

  Returns
  -------
  tf.Tensor
    A 1-D tensor of length ``N``.

  Raises
  ------
  InvalidArgumentError
    If the inputs have Inf or NaN values.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg='')]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)

  if len(x.get_shape()) == 1:
    vec = x
    mat = y
    return tf.reshape(tf.matmul(tf.expand_dims(vec, 0), mat), [-1])
  else:
    mat = x
    vec = y
    return tf.reshape(tf.matmul(mat, tf.expand_dims(vec, 1)), [-1])


class Empty(object):
  """Empty class."""
  pass


def get_dims(x):
  """Get values of each dimension.

  Parameters
  ----------
  x : float, int, tf.Tensor, np.ndarray, or RandomVariable
    A n-D tensor.

  Returns
  -------
  list of int
    Python list containing dimensions of ``x``.
  """
  if isinstance(x, float) or isinstance(x, int):
    return []
  elif isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
    return x.get_shape().as_list()
  elif isinstance(x, np.ndarray):
    return list(x.shape)
  elif isinstance(x, RandomVariable):
    return x.get_batch_shape().as_list()
  else:
    raise NotImplementedError()


def get_session():
  """Get the globally defined TensorFlow session.

  If the session is not already defined, then the function will create
  a global session.

  Returns
  -------
  _ED_SESSION : tf.InteractiveSession
  """
  global _ED_SESSION
  if tf.get_default_session() is None:
    _ED_SESSION = tf.InteractiveSession()
  else:
    _ED_SESSION = tf.get_default_session()

  return _ED_SESSION


def hessian(y, xs):
  """Calculate Hessian of y with respect to each x in xs.

  Parameters
  ----------
  y : tf.Tensor
    Tensor to calculate Hessian of.
  xs : list of tf.Variable
    List of TensorFlow variables to calculate with respect to.
    The variables can have different shapes.

  Returns
  -------
  tf.Tensor
    A 2-D tensor where each row is
    .. math:: \partial_{xs} ( [ \partial_{xs} y ]_j ).

  Raises
  ------
  InvalidArgumentError
    If the inputs have Inf or NaN values.
  """
  y = tf.convert_to_tensor(y)
  dependencies = [tf.verify_tensor_all_finite(y, msg='')]
  dependencies.extend([tf.verify_tensor_all_finite(x, msg='') for x in xs])

  with tf.control_dependencies(dependencies):
    # Calculate flattened vector grad_{xs} y.
    grads = tf.gradients(y, xs)
    grads = [tf.reshape(grad, [-1]) for grad in grads]
    grads = tf.concat(0, grads)
    # Loop over each element in the vector.
    mat = []
    d = grads.get_shape()[0]
    if not isinstance(d, int):
      d = grads.eval().shape[0]

    for j in range(d):
      # Calculate grad_{xs} ( [ grad_{xs} y ]_j ).
      gradjgrads = tf.gradients(grads[j], xs)
      # Flatten into vector.
      hi = []
      for l in range(len(xs)):
        hij = gradjgrads[l]
        # return 0 if gradient doesn't exist; TensorFlow returns None
        if hij is None:
          hij = tf.zeros(xs[l].get_shape(), dtype=tf.float32)

        hij = tf.reshape(hij, [-1])
        hi.append(hij)

      hi = tf.concat(0, hi)
      mat.append(hi)

    # Form matrix where each row is grad_{xs} ( [ grad_{xs} y ]_j ).
    return tf.pack(mat)


def kl_multivariate_normal(loc_one, scale_one, loc_two=0.0, scale_two=1.0):
  """Calculate the KL of multivariate normal distributions with
  diagonal covariances.

  Parameters
  ----------
  loc_one : tf.Tensor
    A 0-D tensor, 1-D tensor of length n, or 2-D tensor of shape M
    x n where each row represents the mean of a n-dimensional
    Gaussian.
  scale_one : tf.Tensor
    A tensor of same shape as ``loc_one``, representing the
    standard deviation.
  loc_two : tf.Tensor, optional
    A tensor of same shape as ``loc_one``, representing the
    mean of another Gaussian.
  scale_two : tf.Tensor, optional
    A tensor of same shape as ``loc_one``, representing the
    standard deviation of another Gaussian.

  Returns
  -------
  tf.Tensor
    For 0-D or 1-D tensor inputs, outputs the 0-D tensor
    ``KL( N(z; loc_one, scale_one) || N(z; loc_two, scale_two) )``
    For 2-D tensor inputs, outputs the 1-D tensor
    ``[KL( N(z; loc_one[m,:], scale_one[m,:]) || ``
    ``N(z; loc_two[m,:], scale_two[m,:]) )]_{m=1}^M``

  Raises
  ------
  InvalidArgumentError
    If the location variables have Inf or NaN values, or if the scale
    variables are not positive.
  """
  loc_one = tf.convert_to_tensor(loc_one)
  scale_one = tf.convert_to_tensor(scale_one)
  loc_two = tf.convert_to_tensor(loc_two)
  scale_two = tf.convert_to_tensor(scale_two)
  dependencies = [tf.verify_tensor_all_finite(loc_one, msg=''),
                  tf.verify_tensor_all_finite(loc_two, msg=''),
                  tf.assert_positive(scale_one),
                  tf.assert_positive(scale_two)]
  loc_one = control_flow_ops.with_dependencies(dependencies, loc_one)
  scale_one = control_flow_ops.with_dependencies(dependencies, scale_one)

  if loc_two == 0.0 and scale_two == 1.0:
    # With default arguments, we can avoid some intermediate computation.
    out = tf.square(scale_one) + tf.square(loc_one) - \
        1.0 - 2.0 * tf.log(scale_one)
  else:
    loc_two = control_flow_ops.with_dependencies(dependencies, loc_two)
    scale_two = control_flow_ops.with_dependencies(dependencies, scale_two)
    out = tf.square(scale_one / scale_two) + \
        tf.square((loc_two - loc_one) / scale_two) - \
        1.0 + 2.0 * tf.log(scale_two) - 2.0 * tf.log(scale_one)

  if len(out.get_shape()) <= 1:  # scalar or vector
    return 0.5 * tf.reduce_sum(out)
  else:  # matrix
    return 0.5 * tf.reduce_sum(out, 1)


def log_mean_exp(input_tensor, reduction_indices=None, keep_dims=False):
  """Compute the ``log_mean_exp`` of elements in a tensor, taking
  the mean across axes given by ``reduction_indices``.

  Parameters
  ----------
  input_tensor : tf.Tensor
    The tensor to reduce. Should have numeric type.
  reduction_indices : int or list of int, optional
    The dimensions to reduce. If `None` (the default), reduces all
    dimensions.
  keep_dims : bool, optional
    If true, retains reduced dimensions with length 1.

  Returns
  -------
  tf.Tensor
    The reduced tensor.

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  dependencies = [tf.verify_tensor_all_finite(input_tensor, msg='')]
  input_tensor = control_flow_ops.with_dependencies(dependencies, input_tensor)

  x_max = tf.reduce_max(input_tensor, reduction_indices, keep_dims=True)
  return tf.squeeze(x_max) + tf.log(tf.reduce_mean(
      tf.exp(input_tensor - x_max), reduction_indices, keep_dims))


def log_sum_exp(input_tensor, reduction_indices=None, keep_dims=False):
  """Compute the ``log_sum_exp`` of elements in a tensor, taking
  the sum across axes given by ``reduction_indices``.

  Parameters
  ----------
  input_tensor : tf.Tensor
    The tensor to reduce. Should have numeric type.
  reduction_indices : int or list of int, optional
    The dimensions to reduce. If `None` (the default), reduces all
    dimensions.
  keep_dims : bool, optional
    If true, retains reduced dimensions with length 1.

  Returns
  -------
  tf.Tensor
    The reduced tensor.

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  dependencies = [tf.verify_tensor_all_finite(input_tensor, msg='')]
  input_tensor = control_flow_ops.with_dependencies(dependencies, input_tensor)

  x_max = tf.reduce_max(input_tensor, reduction_indices, keep_dims=True)
  return tf.squeeze(x_max) + tf.log(tf.reduce_sum(
      tf.exp(input_tensor - x_max), reduction_indices, keep_dims))


def logit(x):
  """Evaluate :math:`\log(x / (1 - x))` elementwise.

  Parameters
  ----------
  x : tf.Tensor
    A n-D tensor.

  Returns
  -------
  tf.Tensor
    A tensor of same shape as input.

  Raises
  ------
  InvalidArgumentError
    If the input is not between :math:`(0,1)` elementwise.
  """
  x = tf.convert_to_tensor(x)
  dependencies = [tf.assert_positive(x),
                  tf.assert_less(x, 1.0)]
  x = control_flow_ops.with_dependencies(dependencies, x)

  return tf.log(x) - tf.log(1.0 - x)


def multivariate_rbf(x, y=0.0, sigma=1.0, l=1.0):
  """Squared-exponential kernel

  .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) \sum_i (x_i - y_i)^2 }

  Parameters
  ----------
  x : tf.Tensor
    A n-D tensor.
  y : tf.Tensor, optional
    A tensor of same shape as ``x``.
  sigma : tf.Tensor, optional
    A 0-D tensor, representing the standard deviation of radial
    basis function.
  l : tf.Tensor, optional
    A 0-D tensor, representing the lengthscale of radial basis
    function.

  Returns
  -------
  tf.Tensor
    A tensor of one less dimension than the input.

  Raises
  ------
  InvalidArgumentError
    If the mean variables have Inf or NaN values, or if the scale
    and length variables are not positive.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  sigma = tf.convert_to_tensor(sigma)
  l = tf.convert_to_tensor(l)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg=''),
                  tf.assert_positive(sigma),
                  tf.assert_positive(l)]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)
  sigma = control_flow_ops.with_dependencies(dependencies, sigma)
  l = control_flow_ops.with_dependencies(dependencies, l)

  return tf.pow(sigma, 2.0) * \
      tf.exp(-1.0 / (2.0 * tf.pow(l, 2.0)) * tf.reduce_sum(tf.pow(x - y, 2.0)))


def placeholder(*args, **kwargs):
  """A wrapper around ``tf.placeholder``. It adds the tensor to the
  ``PLACEHOLDERS`` collection."""
  x = tf.placeholder(*args, **kwargs)
  tf.add_to_collection("PLACEHOLDERS", x)
  return x


def rbf(x, y=0.0, sigma=1.0, l=1.0):
  """Squared-exponential kernel element-wise

  .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) (x - y)^2 }

  Parameters
  ----------
  x : tf.Tensor
    A n-D tensor.
  y : tf.Tensor, optional
    A tensor of same shape as ``x``.
  sigma : tf.Tensor, optional
    A 0-D tensor, representing the standard deviation of radial
    basis function.
  l : tf.Tensor, optional
    A 0-D tensor, representing the lengthscale of radial basis
    function.

  Returns
  -------
  tf.Tensor
    A tensor of one less dimension than the input.

  Raises
  ------
  InvalidArgumentError
    If the mean variables have Inf or NaN values, or if the scale
    and length variables are not positive.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  sigma = tf.convert_to_tensor(sigma)
  l = tf.convert_to_tensor(l)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg=''),
                  tf.assert_positive(sigma),
                  tf.assert_positive(l)]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)
  sigma = control_flow_ops.with_dependencies(dependencies, sigma)
  l = control_flow_ops.with_dependencies(dependencies, l)

  return tf.pow(sigma, 2.0) * \
      tf.exp(-1.0 / (2.0 * tf.pow(l, 2.0)) * tf.pow(x - y, 2.0))


def set_seed(x):
  """Set seed for both NumPy and TensorFlow.

  Parameters
  ----------
  x : int, float
    seed
  """
  node_names = list(six.iterkeys(tf.get_default_graph()._nodes_by_name))
  if len(node_names) > 0 and node_names != ['keras_learning_phase']:
    raise RuntimeError("Seeding is not supported after initializing "
                       "part of the graph. "
                       "Please move set_seed to the beginning of your code.")

  np.random.seed(x)
  tf.set_random_seed(x)


def tile(input, multiples, *args, **kwargs):
  """Constructs a tensor by tiling a given tensor.

  This extends ``tf.tile`` to features available in ``np.tile``.
  Namely, ``inputs`` and ``multiples`` can be a 0-D tensor.  Further,
  if 1-D, ``multiples`` can be of any length according to broadcasting
  rules (see documentation of ``np.tile`` or examples below).

  Parameters
  ----------
  input : tf.Tensor
    The input tensor.
  multiples : tf.Tensor
    The number of repetitions of ``input`` along each axis. Has type
    ``tf.int32``. 0-D or 1-D.
  *args :
    Passed into ``tf.tile``.
  **kwargs :
    Passed into ``tf.tile``.

  Returns
  -------
  tf.Tensor
      Has the same type as ``input``.

  Examples
  --------
  >>> a = tf.constant([0, 1, 2])
  >>> sess.run(ed.tile(a, 2))
  array([0, 1, 2, 0, 1, 2], dtype=int32)
  >>> sess.run(ed.tile(a, (2, 2)))
  array([[0, 1, 2, 0, 1, 2],
         [0, 1, 2, 0, 1, 2]], dtype=int32)
  >>> sess.run(ed.tile(a, (2, 1, 2)))
  array([[[0, 1, 2, 0, 1, 2]],
         [[0, 1, 2, 0, 1, 2]]], dtype=int32)
  >>>
  >>> b = tf.constant([[1, 2], [3, 4]])
  >>> sess.run(ed.tile(b, 2))
  array([[1, 2, 1, 2],
         [3, 4, 3, 4]], dtype=int32)
  >>> sess.run(ed.tile(b, (2, 1)))
  array([[1, 2],
         [3, 4],
         [1, 2],
         [3, 4]], dtype=int32)
  >>>
  >>> c = tf.constant([1, 2, 3, 4])
  >>> sess.run(ed.tile(c, (4, 1)))
  array([[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]], dtype=int32)

  Notes
  -----
  Sometimes this can result in an unknown shape. The core reason for
  this is the following behavior:

  >>> n = tf.constant([1])
  >>> tf.tile(tf.constant([[1.0]]),
  ...         tf.concat(0, [n, tf.constant([1.0]).get_shape()]))
  <tf.Tensor 'Tile:0' shape=(1, 1) dtype=float32>
  >>> n = tf.reshape(tf.constant(1), [1])
  >>> tf.tile(tf.constant([[1.0]]),
  ...         tf.concat(0, [n, tf.constant([1.0]).get_shape()]))
  <tf.Tensor 'Tile_1:0' shape=(?, ?) dtype=float32>

  For this reason, we try to fetch ``multiples`` out of session if
  possible. This can be slow if ``multiples`` has computationally
  intensive dependencies in order to perform this fetch.
  """
  input = tf.convert_to_tensor(input)
  multiples = tf.convert_to_tensor(multiples)

  # 0-d tensor
  if len(input.get_shape()) == 0:
    input = tf.expand_dims(input, 0)

  # 0-d tensor
  if len(multiples.get_shape()) == 0:
    multiples = tf.expand_dims(multiples, 0)

  try:
    get_session()
    multiples = tf.convert_to_tensor(multiples.eval())
  except:
    pass

  # broadcasting
  diff = len(input.get_shape()) - get_dims(multiples)[0]
  if diff < 0:
    input = tf.reshape(input, [1] * np.abs(diff) + get_dims(input))
  elif diff > 0:
    multiples = tf.concat(0, [tf.ones(diff, dtype=tf.int32), multiples])

  return tf.tile(input, multiples, *args, **kwargs)


def to_simplex(x):
  """Transform real vector of length ``(K-1)`` to a simplex of dimension ``K``
  using a backward stick breaking construction.

  Parameters
  ----------
  x : tf.Tensor
    A 1-D or 2-D tensor.

  Returns
  -------
  tf.Tensor
    A tensor of same shape as input but with last dimension of
    size ``K``.

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.

  Notes
  -----
  x as a 3-D or higher tensor is not guaranteed to be supported.
  """
  x = tf.cast(x, dtype=tf.float32)
  dependencies = [tf.verify_tensor_all_finite(x, msg='')]
  x = control_flow_ops.with_dependencies(dependencies, x)

  if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
    shape = get_dims(x)
  else:
    shape = x.shape

  if len(shape) == 1:
    n_rows = ()
    K_minus_one = shape[0]
    eq = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one), dtype=tf.float32))
    z = tf.sigmoid(eq + x)
    pil = tf.concat(0, [z, tf.constant([1.0])])
    piu = tf.concat(0, [tf.constant([1.0]), 1.0 - z])
    S = cumprod(piu)
    return S * pil
  else:
    n_rows = shape[0]
    K_minus_one = shape[1]
    eq = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one), dtype=tf.float32))
    z = tf.sigmoid(eq + x)
    pil = tf.concat(1, [z, tf.ones([n_rows, 1])])
    piu = tf.concat(1, [tf.ones([n_rows, 1]), 1.0 - z])
    # cumulative product along 1st axis
    S = tf.pack([cumprod(piu_x) for piu_x in tf.unpack(piu)])
    return S * pil
