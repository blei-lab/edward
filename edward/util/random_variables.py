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


def check_data(data):
  """Check that the data dictionary passed during inference and
  criticism is valid.
  """
  if not isinstance(data, dict):
    raise TypeError("data must have type dict.")

  for key, value in six.iteritems(data):
    if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
      if isinstance(value, RandomVariable):
        raise TypeError("The value of a feed cannot be a ed.RandomVariable "
                        "object. "
                        "Acceptable feed values include Python scalars, "
                        "strings, lists, numpy ndarrays, or TensorHandles.")
      elif isinstance(value, tf.Tensor):
        raise TypeError("The value of a feed cannot be a tf.Tensor object. "
                        "Acceptable feed values include Python scalars, "
                        "strings, lists, numpy ndarrays, or TensorHandles.")
    elif isinstance(key, (RandomVariable, tf.Tensor)):
      if isinstance(value, (RandomVariable, tf.Tensor)):
        if not key.shape.is_compatible_with(value.shape):
          raise TypeError("Key-value pair in data does not have same "
                          "shape: {}, {}".format(key.shape, value.shape))
        elif key.dtype != value.dtype:
          raise TypeError("Key-value pair in data does not have same "
                          "dtype: {}, {}".format(key.dtype, value.dtype))
      elif isinstance(value, (float, list, int, np.ndarray, np.number, str)):
        if not key.shape.is_compatible_with(np.shape(value)):
          raise TypeError("Key-value pair in data does not have same "
                          "shape: {}, {}".format(key.shape, np.shape(value)))
        elif isinstance(value, (np.ndarray, np.number)) and \
                not np.issubdtype(value.dtype, np.float) and \
                not np.issubdtype(value.dtype, np.int) and \
                not np.issubdtype(value.dtype, np.str):
          raise TypeError("Data value has an invalid dtype: "
                          "{}".format(value.dtype))
      else:
        raise TypeError("Data value has an invalid type: "
                        "{}".format(type(value)))
    else:
      raise TypeError("Data key has an invalid type: {}".format(type(key)))


def check_latent_vars(latent_vars):
  """Check that the latent variable dictionary passed during inference and
  criticism is valid.
  """
  if not isinstance(latent_vars, dict):
    raise TypeError("latent_vars must have type dict.")

  for key, value in six.iteritems(latent_vars):
    if not isinstance(key, (RandomVariable, tf.Tensor)):
      raise TypeError("Latent variable key has an invalid type: "
                      "{}".format(type(key)))
    elif not isinstance(value, (RandomVariable, tf.Tensor)):
      raise TypeError("Latent variable value has an invalid type: "
                      "{}".format(type(value)))
    elif not key.shape.is_compatible_with(value.shape):
      raise TypeError("Key-value pair in latent_vars does not have same "
                      "shape: {}, {}".format(key.shape, value.shape))
    elif key.dtype != value.dtype:
      raise TypeError("Key-value pair in latent_vars does not have same "
                      "dtype: {}, {}".format(key.dtype, value.dtype))


def copy_default(x, *args, **kwargs):
  if isinstance(x, (RandomVariable, tf.Operation, tf.Tensor, tf.Variable)):
    x = copy(x, *args, **kwargs)

  return x


def copy(org_instance, dict_swap=None, scope="copied",
         replace_itself=False, copy_q=False):
  """Build a new node in the TensorFlow graph from `org_instance`,
  where any of its ancestors existing in `dict_swap` are
  replaced with `dict_swap`'s corresponding value.

  Copying is done recursively. Any `Operation` whose output is
  required to copy `org_instance` is also copied (if it isn't already
  copied within the new scope).

  `tf.Variable`s, `tf.placeholder`s, and nodes of type `Queue` are
  always reused and not copied. In addition, `tf.Operation`s with
  operation-level seeds are copied with a new operation-level seed.

  Parameters
  ----------
  org_instance : RandomVariable, tf.Operation, tf.Tensor, or tf.Variable
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
  >>> # `qx` -> `copied/z` <- `copied/y`
  >>> z_new = ed.copy(z, {x: qx})
  >>>
  >>> sess = tf.Session()
  >>> sess.run(z)
  6.0
  >>> sess.run(z_new)
  12.0
  """
  if not isinstance(org_instance,
                    (RandomVariable, tf.Operation, tf.Tensor, tf.Variable)):
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

  # If instance is a tf.Variable, return it; do not copy any. Note we
  # check variables via their name. If we get variables through an
  # op's inputs, it has type tf.Tensor and not tf.Variable.
  if isinstance(org_instance, (tf.Tensor, tf.Variable)):
    for variable in tf.global_variables():
      if org_instance.name == variable.name:
        if variable in dict_swap and replace_itself:
          # Deal with case when `org_instance` is the associated _ref
          # tensor for a tf.Variable.
          org_instance = dict_swap[variable]
          if not copy_q or isinstance(org_instance, tf.Variable):
            return org_instance
          for variable in tf.global_variables():
            if org_instance.name == variable.name:
              return variable
          break
        else:
          return variable

  graph = tf.get_default_graph()
  new_name = scope + '/' + org_instance.name

  # If an instance of the same name exists, return it.
  if isinstance(org_instance, RandomVariable):
    for rv in random_variables():
      if new_name == rv.unique_name:
        return rv
  elif isinstance(org_instance, (tf.Tensor, tf.Operation)):
    try:
      return graph.as_graph_element(new_name,
                                    allow_tensor=True,
                                    allow_operation=True)
    except:
      pass

  if isinstance(org_instance, RandomVariable):
    rv = org_instance

    # If it has copiable arguments, copy them.
    args = [copy_default(arg, dict_swap, scope, True, copy_q)
            for arg in rv._args]

    kwargs = {}
    for key, value in six.iteritems(rv._kwargs):
      if isinstance(value, list):
        kwargs[key] = [copy_default(v, dict_swap, scope, True, copy_q)
                       for v in value]
      else:
        kwargs[key] = copy_default(value, dict_swap, scope, True, copy_q)

    kwargs['name'] = new_name
    # Create new random variable with copied arguments.
    new_rv = type(rv)(*args, **kwargs)
    return new_rv
  elif isinstance(org_instance, tf.Tensor):
    tensor = org_instance

    # Do not copy tf.placeholders.
    if 'Placeholder' in tensor.op.type:
      return tensor

    # A tensor is one of the outputs of its underlying
    # op. Therefore copy the op itself.
    op = tensor.op
    new_op = copy(op, dict_swap, scope, True, copy_q)

    output_index = op.outputs.index(tensor)
    new_tensor = new_op.outputs[output_index]

    # Add copied tensor to collections that the original one is in.
    for name, collection in six.iteritems(tensor.graph._collections):
      if tensor in collection:
        graph.add_to_collection(name, new_tensor)

    return new_tensor
  elif isinstance(org_instance, tf.Operation):
    op = org_instance

    # Do not copy queue operations.
    if 'Queue' in op.type:
      return op

    # Copy the node def.
    # It is unique to every Operation instance. Replace the name and
    # its operation-level seed if it has one.
    node_def = deepcopy(op.node_def)
    node_def.name = new_name
    if 'seed2' in node_def.attr and tf.get_seed(None)[1] is not None:
      node_def.attr['seed2'].i = tf.get_seed(None)[1]

    # Copy other arguments needed for initialization.
    output_types = op._output_types[:]

    # If it has an original op, copy it.
    if op._original_op is not None:
      original_op = copy(op._original_op, dict_swap, scope, True, copy_q)
    else:
      original_op = None

    # Copy the op def.
    # It is unique to every Operation type.
    op_def = deepcopy(op.op_def)

    new_op = tf.Operation(node_def,
                          graph,
                          [],  # inputs; will add them afterwards
                          output_types,
                          [],  # control inputs; will add them afterwards
                          [],  # input types; will add them afterwards
                          original_op,
                          op_def)

    # advertise op early to break recursions
    graph._add_op(new_op)

    # If it has control inputs, copy them.
    control_inputs = []
    for x in op.control_inputs:
      elem = copy(x, dict_swap, scope, True, copy_q)
      if not isinstance(elem, tf.Operation):
        elem = tf.convert_to_tensor(elem)

      control_inputs.append(elem)

    new_op._add_control_inputs(control_inputs)

    # If it has inputs, copy them.
    for x in op.inputs:
      elem = copy(x, dict_swap, scope, True, copy_q)
      if not isinstance(elem, tf.Operation):
        elem = tf.convert_to_tensor(elem)

      new_op._add_input(elem)

    # Use Graph's private methods to add the op, following
    # implementation of `tf.Graph().create_op()`.
    compute_shapes = True
    compute_device = True
    op_type = new_name

    if compute_shapes:
      set_shapes_for_outputs(new_op)
    graph._record_op_seen_by_control_dependencies(new_op)

    if compute_device:
      graph._apply_device_functions(new_op)

    if graph._colocation_stack:
      all_colocation_groups = []
      for colocation_op in graph._colocation_stack:
        all_colocation_groups.extend(colocation_op.colocation_groups())
        if colocation_op.device:
          # Make this device match the device of the colocated op, to
          # provide consistency between the device and the colocation
          # property.
          if new_op.device and new_op.device != colocation_op.device:
            logging.warning("Tried to colocate %s with an op %s that had "
                            "a different device: %s vs %s. "
                            "Ignoring colocation property.",
                            name, colocation_op.name, new_op.device,
                            colocation_op.device)
          else:
            new_op._set_device(colocation_op.device)

      all_colocation_groups = sorted(set(all_colocation_groups))
      new_op.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))

    # Sets "container" attribute if
    # (1) graph._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if (graph._container and
        op_type in graph._registered_ops and
        graph._registered_ops[op_type].is_stateful and
        "container" in new_op.node_def.attr and
            not new_op.node_def.attr["container"].s):
      new_op.node_def.attr["container"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(graph._container)))

    return new_op
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
  >>> a = Normal(0.0, 1.0)
  >>> b = Normal(a, 1.0)
  >>> c = Normal(0.0, 1.0)
  >>> d = Normal(b * c, 1.0)
  >>> assert set(ed.get_ancestors(d)) == set([a, b, c])
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)

    nodes.update(node.op.inputs)

  return list(output)


def get_blanket(x, collection=None):
  """Get Markov blanket of input, which consists of its parents, its
  children, and the other parents of its children.

  Parameters
  ----------
  x : RandomVariable or tf.Tensor
    Query node to find Markov blanket of.
  collection : list of RandomVariable, optional
    The collection of random variables to check with respect to;
    defaults to all random variables in the graph.

  Returns
  -------
  list of RandomVariable
    Markov blanket of x.

  Examples
  --------
  >>> a = Normal(0.0, 1.0)
  >>> b = Normal(0.0, 1.0)
  >>> c = Normal(a * b, 1.0)
  >>> d = Normal(0.0, 1.0)
  >>> e = Normal(c * d, 1.0)
  >>> assert set(ed.get_blanket(c)) == set([a, b, d, e])
  """
  output = set()
  output.update(get_parents(x, collection))
  children = get_children(x, collection)
  output.update(children)
  for child in children:
    output.update(get_parents(child, collection))

  output.discard(x)
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
  >>> a = Normal(0.0, 1.0)
  >>> b = Normal(a, 1.0)
  >>> c = Normal(a, 1.0)
  >>> d = Normal(c, 1.0)
  >>> assert set(ed.get_children(a)) == set([b, c])
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
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
  >>> a = Normal(0.0, 1.0)
  >>> b = Normal(a, 1.0)
  >>> c = Normal(a, 1.0)
  >>> d = Normal(c, 1.0)
  >>> assert set(ed.get_descendants(a)) == set([b, c, d])
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)

    for op in node.consumers():
      nodes.update(op.outputs)

  return list(output)


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
  >>> a = Normal(0.0, 1.0)
  >>> b = Normal(a, 1.0)
  >>> c = Normal(0.0, 1.0)
  >>> d = Normal(b * c, 1.0)
  >>> assert set(ed.get_parents(d)) == set([b, c])
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value(): node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
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
  >>> a = Normal(0.0, 1.0)
  >>> b = Normal(a, 1.0)
  >>> c = Normal(a, 1.0)
  >>> assert ed.get_siblings(b) == [c]
  """
  parents = get_parents(x, collection)
  siblings = set()
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
  >>> c = Normal(a * b, 1.0)
  >>> assert set(ed.get_variables(c)) == set([a, b])
  """
  if collection is None:
    collection = tf.global_variables()

  node_dict = {node.name: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value()

    candidate_node = node_dict.get(node.name, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)

    nodes.update(node.op.inputs)

  return list(output)
