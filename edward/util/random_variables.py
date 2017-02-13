from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from copy import deepcopy
from edward.models.random_variable import RandomVariable
from edward.util.graphs import random_variables
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework.ops import set_shapes_for_outputs
from tensorflow.python.util import compat


def copy_rv(value, dict_swap, scope, replace_itself, copy_q):
  if isinstance(value, RandomVariable) or \
     isinstance(value, tf.Variable) or \
     isinstance(value, tf.Tensor) or \
     isinstance(value, tf.Operation):
      value = copy(value, dict_swap, scope, replace_itself, copy_q)
  return value


def copy(org_instance, dict_swap=None, scope="copied",
         replace_itself=False, copy_q=False):
  """Build a new node in the TensorFlow graph from `org_instance`,
  where any of its ancestors existing in `dict_swap` are
  replaced with `dict_swap`'s corresponding value.

  The copying is done recursively, so any `Operation` whose output
  is required to evaluate `org_instance` is also copied (if it isn't
  already copied within the new scope). This is with the exception of
  `tf.Variable`s, `tf.placeholder`s, and nodes of type `Queue`, which
  are reused and not newly copied.

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
  variables = {x.name: x for
               x in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
  if org_instance.name in variables:
    return graph.get_tensor_by_name(variables[org_instance.name].name)

  # Do the same for placeholders. Determine via its op's type.
  if isinstance(org_instance, tf.Tensor):
    if "Placeholder" in org_instance.op.type:
      return org_instance

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
      if isinstance(value, list):
        kwargs[key] = [
            copy_rv(v, dict_swap, scope, True, copy_q) for v in value
        ]
      else:
        kwargs[key] = copy_rv(value, dict_swap, scope, True, copy_q)

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
  elif isinstance(org_instance, tf.Operation):
    op = org_instance

    # Do not copy queue operations
    if 'Queue' in op.type:
      return op

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
  else:
    raise TypeError("Could not copy instance: " + str(org_instance))


def get_ancestors(x, collection=None):
  """Get ancestor random variables of input.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find ancestors of.
  collection : list of RandomVariable, optional
    The collection of random variables to check with respect to;
    defaults to all random variables in the graph.

  Returns
  -------
  list of RandomVariable
    Ancestor random variables of x.

  Examples
  --------
  >>> a = Normal(mu=0.0, sigma=1.0)
  >>> b = Normal(mu=a, sigma=1.0)
  >>> c = Normal(mu=0.0, sigma=1.0)
  >>> d = Normal(mu=tf.mul(b, c), sigma=1.0)
  >>> set(ed.get_ancestors(d)) == set([a, b, c])
  True
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set([])
  nodes = set([x])
  while nodes:
    node = nodes.pop()
    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node and candidate_node != x:
      output.add(candidate_node)

    nodes.update(node.op.inputs)

  return list(output)


def get_children(x, collection=None):
  """Get child random variables of input.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find children of.
  collection : list of RandomVariable, optional
    The collection of random variables to check with respect to;
    defaults to all random variables in the graph.

  Returns
  -------
  list of RandomVariable
    Child random variables of x.

  Examples
  --------
  >>> a = Normal(mu=0.0, sigma=1.0)
  >>> b = Normal(mu=a, sigma=1.0)
  >>> c = Normal(mu=a, sigma=1.0)
  >>> d = Normal(mu=c, sigma=1.0)
  >>> set(ed.get_children(a)) == set([b, c])
  True
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set([])
  nodes = set([x])
  while nodes:
    node = nodes.pop()
    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node and candidate_node != x:
      output.add(candidate_node)
    else:
      for op in node.consumers():
        nodes.update(op.outputs)

  return list(output)


def get_descendants(x, collection=None):
  """Get descendant random variables of input.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find descendants of.
  collection : list of RandomVariable, optional
    The collection of random variables to check with respect to;
    defaults to all random variables in the graph.

  Returns
  -------
  list of RandomVariable
    Descendant random variables of x.

  Examples
  --------
  >>> a = Normal(mu=0.0, sigma=1.0)
  >>> b = Normal(mu=a, sigma=1.0)
  >>> c = Normal(mu=a, sigma=1.0)
  >>> d = Normal(mu=c, sigma=1.0)
  >>> set(ed.get_descendants(a)) == set([b, c, d])
  True
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set([])
  nodes = set([x])
  while nodes:
    node = nodes.pop()
    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node and candidate_node != x:
      output.add(candidate_node)

    for op in node.consumers():
      nodes.update(op.outputs)

  return list(output)


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


def get_parents(x, collection=None):
  """Get parent random variables of input.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find parents of.
  collection : list of RandomVariable, optional
    The collection of random variables to check with respect to;
    defaults to all random variables in the graph.

  Returns
  -------
  list of RandomVariable
    Parent random variables of x.

  Examples
  --------
  >>> a = Normal(mu=0.0, sigma=1.0)
  >>> b = Normal(mu=a, sigma=1.0)
  >>> c = Normal(mu=0.0, sigma=1.0)
  >>> d = Normal(mu=tf.mul(b, c), sigma=1.0)
  >>> set(ed.get_parents(d)) == set([b, c])
  True
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set([])
  nodes = set([x])
  while nodes:
    node = nodes.pop()
    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node and candidate_node != x:
      output.add(candidate_node)
    else:
      nodes.update(node.op.inputs)

  return list(output)


def get_siblings(x, collection=None):
  """Get sibling random variables of input.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find siblings of.
  collection : list of RandomVariable, optional
    The collection of random variables to check with respect to;
    defaults to all random variables in the graph.

  Returns
  -------
  list of RandomVariable
    Sibling random variables of x.

  Examples
  --------
  >>> a = Normal(mu=0.0, sigma=1.0)
  >>> b = Normal(mu=a, sigma=1.0)
  >>> c = Normal(mu=a, sigma=1.0)
  >>> ed.get_siblings(b) == [c]
  True
  """
  parents = get_parents(x, collection)
  siblings = set([])
  for parent in parents:
    siblings.update(get_children(parent, collection))

  siblings.discard(x)
  return list(siblings)


def get_variables(x, collection=None):
  """Get parent TensorFlow variables of input.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find parents of.
  collection : list of tf.Variable, optional
    The collection of variables to check with respect to; defaults to
    all variables in the graph.

  Returns
  -------
  list of tf.Variable
    TensorFlow variables that x depends on.

  Examples
  --------
  >>> a = tf.Variable(0.0)
  >>> b = tf.Variable(0.0)
  >>> c = Normal(mu=tf.mul(a, b), sigma=1.0)
  >>> set(ed.get_variables(c)) == set([a, b])
  True
  """
  if collection is None:
    collection = tf.global_variables()

  node_dict = {node.name: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set([])
  nodes = set([x])
  while nodes:
    node = nodes.pop()
    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node.name, None)
    if candidate_node and candidate_node != x:
      output.add(candidate_node)

    nodes.update(node.op.inputs)

  return list(output)
