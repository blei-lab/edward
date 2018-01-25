from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from copy import deepcopy
from edward.models.random_variable import RandomVariable
from edward.models.random_variables import TransformedDistribution
from edward.models import PointMass
from edward.util.graphs import random_variables
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework.ops import set_shapes_for_outputs
from tensorflow.python.util import compat

tfb = tf.contrib.distributions.bijectors


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


def _get_context_copy(ctx, scope):
    # contexts are stored in graph collections
    # is there a more efficient way to do this?

    graph = tf.get_default_graph()

    for name, collection in six.iteritems(graph._collections):
      if ctx in collection:
        for item in collection:
          if item.name == scope + ctx.name:
            return item

    return None


def _copy_context(ctx, context_matches, dict_swap, scope, copy_q):
  if ctx is None:
    return None

  # We'd normally check about returning early, but the context won't
  # be copied until after all children are, so we check that first.

  graph = tf.get_default_graph()

  # copy all nodes within context
  for tensorname in ctx._values:
    tensor = graph.as_graph_element(tensorname)
    copy(tensor, dict_swap, scope, True, copy_q)

  # now make sure we haven't already copied the context we're currently
  # trying to copy (in the course of copying another child)
  ctx_copy = _get_context_copy(ctx, scope)
  if ctx_copy:
    return ctx_copy

  ctx_copy = ctx.from_proto(ctx.to_proto(), scope[:-1])
  outer_copy = _copy_context(ctx.outer_context, context_matches, dict_swap,
                             scope, copy_q)
  ctx_copy._outer_context = outer_copy

  for name, collection in six.iteritems(graph._collections):
      if ctx in collection:
        graph.add_to_collection(name, ctx_copy)
  return ctx_copy


def _copy_default(x, *args, **kwargs):
  if isinstance(x, (RandomVariable, tf.Operation, tf.Tensor, tf.Variable)):
    x = copy(x, *args, **kwargs)

  return x


def copy(org_instance, dict_swap=None, scope="copied",
         replace_itself=False, copy_q=False, copy_parent_rvs=True):
  """Build a new node in the TensorFlow graph from `org_instance`,
  where any of its ancestors existing in `dict_swap` are
  replaced with `dict_swap`'s corresponding value.

  Copying is done recursively. Any `Operation` whose output is
  required to copy `org_instance` is also copied (if it isn't already
  copied within the new scope).

  `tf.Variable`s, `tf.placeholder`s, and nodes of type `Queue` are
  always reused and not copied. In addition, `tf.Operation`s with
  operation-level seeds are copied with a new operation-level seed.

  Args:
    org_instance: RandomVariable, tf.Operation, tf.Tensor, or tf.Variable.
      Node to add in graph with replaced ancestors.
    dict_swap: dict.
      Random variables, variables, tensors, or operations to swap with.
      Its keys are what `org_instance` may depend on, and its values are
      the corresponding object (not necessarily of the same class
      instance, but must have the same type, e.g., float32) that is used
      in exchange.
    scope: str.
      A scope for the new node(s). This is used to avoid name
      conflicts with the original node(s).
    replace_itself: bool.
      Whether to replace `org_instance` itself if it exists in
      `dict_swap`. (This is used for the recursion.)
    copy_q: bool.
      Whether to copy the replaced tensors too (if not already
      copied within the new scope). Otherwise will reuse them.
    copy_parent_rvs:
      Whether to copy parent random variables `org_instance` depends
      on. Otherwise will copy only the sample tensors and not the
      random variable class itself.

  Returns:
    RandomVariable, tf.Variable, tf.Tensor, or tf.Operation.
    The copied node.

  Raises:
    TypeError.
    If `org_instance` is not one of the above types.

  #### Examples

  ```python
  x = tf.constant(2.0)
  y = tf.constant(3.0)
  z = x * y

  qx = tf.constant(4.0)
  # The TensorFlow graph is currently
  # `x` -> `z` <- y`, `qx`

  # This adds a subgraph with newly copied nodes,
  # `qx` -> `copied/z` <- `copied/y`
  z_new = ed.copy(z, {x: qx})

  sess = tf.Session()
  sess.run(z)
  6.0
  sess.run(z_new)
  12.0
  ```
  """
  if not isinstance(org_instance,
                    (RandomVariable, tf.Operation, tf.Tensor, tf.Variable)):
    raise TypeError("Could not copy instance: " + str(org_instance))

  if dict_swap is None:
    dict_swap = {}
  if scope[-1] != '/':
    scope += '/'

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
  new_name = scope + org_instance.name

  # If an instance of the same name exists, return it.
  if isinstance(org_instance, RandomVariable):
    for rv in random_variables():
      if new_name == rv.name:
        return rv
  elif isinstance(org_instance, (tf.Tensor, tf.Operation)):
    try:
      return graph.as_graph_element(new_name,
                                    allow_tensor=True,
                                    allow_operation=True)
    except:
      pass

  # Preserve ordering of random variables. Random variables are always
  # copied first (from parent -> child) before any deterministic
  # operations that depend on them.
  if copy_parent_rvs and \
          isinstance(org_instance, (RandomVariable, tf.Tensor, tf.Variable)):
    for v in get_parents(org_instance):
      copy(v, dict_swap, scope, True, copy_q, True)

  if isinstance(org_instance, RandomVariable):
    rv = org_instance

    # If it has copiable arguments, copy them.
    args = [_copy_default(arg, dict_swap, scope, True, copy_q, False)
            for arg in rv._args]

    kwargs = {}
    for key, value in six.iteritems(rv._kwargs):
      if isinstance(value, list):
        kwargs[key] = [_copy_default(v, dict_swap, scope, True, copy_q, False)
                       for v in value]
      else:
        kwargs[key] = _copy_default(
            value, dict_swap, scope, True, copy_q, False)

    kwargs['name'] = new_name
    # Create new random variable with copied arguments.
    try:
      new_rv = type(rv)(*args, **kwargs)
    except ValueError:
      # Handle case where parameters are copied under absolute name
      # scope. This can cause an error when creating a new random
      # variable as tf.identity name ops are called on parameters ("op
      # with name already exists"). To avoid remove absolute name scope.
      kwargs['name'] = new_name[:-1]
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
    new_op = copy(op, dict_swap, scope, True, copy_q, False)

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

    # when copying control flow contexts,
    # we need to make sure frame definitions are copied
    if 'frame_name' in node_def.attr and node_def.attr['frame_name'].s != b'':
      node_def.attr['frame_name'].s = (scope.encode('utf-8') +
                                       node_def.attr['frame_name'].s)

    if 'seed2' in node_def.attr and tf.get_seed(None)[1] is not None:
      node_def.attr['seed2'].i = tf.get_seed(None)[1]

    # Copy other arguments needed for initialization.
    output_types = op._output_types[:]

    # If it has an original op, copy it.
    if op._original_op is not None:
      original_op = copy(op._original_op, dict_swap, scope, True, copy_q, False)
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
      elem = copy(x, dict_swap, scope, True, copy_q, False)
      if not isinstance(elem, tf.Operation):
        elem = tf.convert_to_tensor(elem)

      control_inputs.append(elem)

    new_op._add_control_inputs(control_inputs)

    # If it has inputs, copy them.
    for x in op.inputs:
      elem = copy(x, dict_swap, scope, True, copy_q, False)
      if not isinstance(elem, tf.Operation):
        elem = tf.convert_to_tensor(elem)

      new_op._add_input(elem)

    # Copy the control flow context.
    control_flow_context = _copy_context(op._get_control_flow_context(), {},
                                         dict_swap, scope, copy_q)
    new_op._set_control_flow_context(control_flow_context)

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

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find ancestors of.
    collection: list of RandomVariable.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Ancestor random variables of x.

  #### Examples
  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(0.0, 1.0)
  d = Normal(b * c, 1.0)
  assert set(ed.get_ancestors(d)) == set([a, b, c])
  ```
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

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find Markov blanket of.
    collection: list of RandomVariable.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Markov blanket of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(0.0, 1.0)
  c = Normal(a * b, 1.0)
  d = Normal(0.0, 1.0)
  e = Normal(c * d, 1.0)
  assert set(ed.get_blanket(c)) == set([a, b, d, e])
  ```
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

  Args:
    x: RandomVariable or tf.Tensor>
      Query node to find children of.
    collection: list of RandomVariable.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Child random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  d = Normal(c, 1.0)
  assert set(ed.get_children(a)) == set([b, c])
  ```
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

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find descendants of.
    collection: list of RandomVariable.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Descendant random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  d = Normal(c, 1.0)
  assert set(ed.get_descendants(a)) == set([b, c, d])
  ```
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

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find parents of.
    collection: list of RandomVariable.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Parent random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(0.0, 1.0)
  d = Normal(b * c, 1.0)
  assert set(ed.get_parents(d)) == set([b, c])
  ```
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

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find siblings of.
    collection: list of RandomVariable.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Sibling random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  assert ed.get_siblings(b) == [c]
  ```
  """
  parents = get_parents(x, collection)
  siblings = set()
  for parent in parents:
    siblings.update(get_children(parent, collection))

  siblings.discard(x)
  return list(siblings)


def get_variables(x, collection=None):
  """Get parent TensorFlow variables of input.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find parents of.
    collection: list of tf.Variable.
      The collection of variables to check with respect to; defaults to
      all variables in the graph.

  Returns:
    list of tf.Variable.
    TensorFlow variables that x depends on.

  #### Examples

  ```python
  a = tf.Variable(0.0)
  b = tf.Variable(0.0)
  c = Normal(a * b, 1.0)
  assert set(ed.get_variables(c)) == set([a, b])
  ```
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


def is_independent(a, b, condition=None):
  """Assess whether a is independent of b given the random variables in
  condition.

  Implemented using the Bayes-Ball algorithm [@schachter1998bayes].

  Args:
    a: RandomVariable or list of RandomVariable.
       Query node(s).
    b: RandomVariable or list of RandomVariable.
       Query node(s).
    condition: RandomVariable or list of RandomVariable.
       Random variable(s) to condition on.

  Returns:
    bool.
    True if a is independent of b given the random variables in condition.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  assert ed.is_independent(b, c, condition=a)
  ```
  """
  if condition is None:
    condition = []
  if not isinstance(a, list):
    a = [a]
  if not isinstance(b, list):
    b = [b]
  if not isinstance(condition, list):
    condition = [condition]
  A = set(a)
  B = set(b)
  condition = set(condition)

  top_marked = set()
  # The Bayes-Ball algorithm will traverse the belief network
  # and add each node that is relevant to B given condition
  # to the set bottom_marked. A and B are conditionally
  # independent if no node in A is in bottom_marked.
  bottom_marked = set()

  schedule = [(node, "child") for node in B]
  while schedule:
    node, came_from = schedule.pop()

    if node not in condition and came_from == "child":
      if node not in top_marked:
        top_marked.add(node)
        for parent in get_parents(node):
          schedule.append((parent, "child"))

      if not isinstance(node, PointMass) and node not in bottom_marked:
        bottom_marked.add(node)
        if node in A:
          return False  # node in A is relevant to B
        for child in get_children(node):
          schedule.append((child, "parent"))

    elif came_from == "parent":
      if node in condition and node not in top_marked:
        top_marked.add(node)
        for parent in get_parents(node):
          schedule.append((parent, "child"))

      elif node not in condition and node not in bottom_marked:
        bottom_marked.add(node)
        if node in A:
          return False  # node in A is relevant to B
        for child in get_children(node):
          schedule.append((child, "parent"))

  return True


def transform(x, *args, **kwargs):
  """Transform a continuous random variable to the unconstrained space.

  `transform` selects among a number of default transformations which
  depend on the support of the provided random variable:

  + $[0, 1]$ (e.g., Beta): Inverse of sigmoid.
  + $[0, \infty)$ (e.g., Gamma): Inverse of softplus.
  + Simplex (e.g., Dirichlet): Inverse of softmax-centered.
  + $(-\infty, \infty)$ (e.g., Normal, MultivariateNormalTriL): None.

  Args:
    x: RandomVariable.
      Continuous random variable to transform.
    *args, **kwargs:
      Arguments to overwrite when forming the `TransformedDistribution`.
      For example, manually specify the transformation by passing in
      the `bijector` argument.

  Returns:
    RandomVariable.
    A `TransformedDistribution` random variable, or the provided random
    variable if no transformation was applied.

  #### Examples

  ```python
  x = Gamma(1.0, 1.0)
  y = ed.transform(x)
  sess = tf.Session()
  sess.run(y)
  -2.2279539
  ```
  """
  if len(args) != 0 or kwargs.get('bijector', None) is not None:
    return TransformedDistribution(x, *args, **kwargs)

  try:
    support = x.support
  except AttributeError as e:
    msg = """'{}' object has no 'support'
             so cannot be transformed.""".format(type(x).__name__)
    raise AttributeError(msg)

  if support == '01':
    bij = tfb.Invert(tfb.Sigmoid())
    new_support = 'real'
  elif support == 'nonnegative':
    bij = tfb.Invert(tfb.Softplus())
    new_support = 'real'
  elif support == 'simplex':
    bij = tfb.Invert(tfb.SoftmaxCentered(event_ndims=1))
    new_support = 'multivariate_real'
  elif support in ('real', 'multivariate_real'):
    return x
  else:
    msg = "'transform' does not handle supports of type '{}'".format(support)
    raise ValueError(msg)

  new_x = TransformedDistribution(x, bij, *args, **kwargs)
  new_x.support = new_support
  return new_x


def compute_multinomial_mode(probs, total_count=1, seed=None):
  """Compute the mode of a Multinomial random variable.

  Args:
    probs: 1-D Numpy array of Multinomial class probabilities
    total_count: integer number of trials in single Multinomial draw
    seed: a Python integer. Used to create a random seed for the
      distribution

  #### Examples

  ```python
  # returns either [2, 2, 1], [2, 1, 2] or [1, 2, 2]
  probs = np.array(3 * [1/3])
  total_count = 5
  compute_multinomial_mode(probs, total_count)

  # returns [3, 2, 0]
  probs = np.array(3 * [1/3])
  total_count = 5
  compute_multinomial_mode(probs, total_count)
  ```
  """
  def softmax(vec):
    numerator = np.exp(vec)
    return numerator / numerator.sum(axis=0)

  random_state = np.random.RandomState(seed)
  mode = np.zeros_like(probs, dtype=np.int32)
  if total_count == 1:
    mode[np.argmax(probs)] += 1
    return list(mode)
  remaining_count = total_count
  uniform_prob = 1 / total_count

  while remaining_count > 0:
    if (probs < uniform_prob).all():
      probs = softmax(probs)
    mask = probs >= uniform_prob
    overflow_count = int(mask.sum() - remaining_count)
    if overflow_count > 0:
      hot_indices = np.where(mask)[0]
      cold_indices = random_state.choice(hot_indices, overflow_count,
                                         replace=False)
      mask[cold_indices] = False
    mode[mask] += 1
    probs[mask] -= uniform_prob
    remaining_count -= np.sum(mask)
  return mode
