from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.models.random_variable import RandomVariable, random_variables


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

  node_dict = {node.value: node for node in collection}

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
      node = node.value

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

  node_dict = {node.value: node for node in collection}

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
      node = node.value

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

  node_dict = {node.value: node for node in collection}

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
      node = node.value

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

  node_dict = {node.value: node for node in collection}

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
      node = node.value

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
      node = node.value

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

      # TODO
      from edward.models import PointMass
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


del random_variables
