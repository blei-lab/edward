from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class NodeWrapper:
  def __init__(self, node):
    self.node = node

  def __getitem__(self, key):
    return self.node.op.type[key]

  def __str__(self):
    return str(self.node)


def _mul_n(x):
  if len(x) == 2:
    return tf.mul(x[0], x[1])
  else:
    return tf.mul(x[0], _mul_n(x[1:]))


_extractable_nodes = {
  'Add': tf.add_n,
  'Mul': _mul_n,
  'Log': tf.log,
  'Pow': tf.pow,
  'Square': tf.square,
  'Reciprocal': tf.reciprocal,
  'Sqrt': tf.sqrt,
  'Identity': lambda x: x
}
def symbolic_suff_stat(node, base_node, stop_nodes):
  if node == base_node:
    return ('#x',)
  elif node in stop_nodes:
    return (NodeWrapper(node),)

  if node.op.type in _extractable_nodes:
    result = ['#%s' % node.op.type]
  else:
    result = [NodeWrapper(node)]

  result += [symbolic_suff_stat(i, base_node, stop_nodes)
             for i in node.op.inputs]
  return tuple(result)


def as_float(x):
  try:
    result = float(x)
    return result
  except:
    return None


def reconstruct_expr(expr):
  float_val = as_float(expr[0])
  if float_val is not None:
    return float_val
  args = [reconstruct_expr(i) for i in expr[1:]]
  if expr[0][:4] == '#Pow':
    return tf.pow(args[0], np.float32(expr[0][4:]))
  if expr[0][0] == '#':
    tf_fn = _extractable_nodes.get(expr[0][1:], None)
    assert(tf_fn is not None)
    return tf_fn(*args)
  assert(len(expr) == 1)
  assert(isinstance(expr[0], NodeWrapper))
  return expr[0].node



_simplify_fns = []
def simplify(expr, simplify_fns=_simplify_fns):
  while True:
    did_something = False
    for fn in simplify_fns:
      did_something_i, expr = fn(expr)
      did_something = did_something or did_something_i
    if not did_something:
      break
  return expr


def _register_simplify_fn(fn):
  '''Wraps and registers simplification functions.

  A simplification function takes as input an expression and possible
  some other args/kwargs, and returns either None (if it did not find
  anything to do at this node) or a simplified version of the graph
  below this node.

  The wrapped function will repeatedly apply this simplification
  function to all nodes of the graph until it stops doing anything.
  '''
  def wrapped(expr, *args, **kwargs):
    result = fn(expr, *args, **kwargs)
    if result is None:
      did_something = False
      new_args = []
      for i in expr[1:]:
        did_something_i, new_arg = wrapped(i)
        did_something = did_something or did_something_i
        new_args.append(new_arg)
      return did_something, (expr[0],) + tuple(new_args)
    else:
      return True, result

  def repeat_wrapped(expr, *args, **kwargs):
    did_something = False
    did_something_i = True
    while did_something_i:
      did_something_i, expr = wrapped(expr, *args, **kwargs)
      did_something = did_something or did_something_i
    return did_something, expr
    
  _simplify_fns.append(repeat_wrapped)
  return repeat_wrapped


_power_ops = {
  '#Reciprocal': -1.,
  '#Square': 2.,
  '#Sqrt': 0.5,
}
@_register_simplify_fn
def power_op_simplify(expr):
  op_power = _power_ops.get(expr[0], None)
  if op_power:
    return ('#Pow%.4e' % op_power,) + expr[1:]


@_register_simplify_fn
def pow_simplify(expr):
  if expr[0][:4] != '#Pow':
    return None
  if expr[1][0][:4] != '#Pow':
    return None

  op_power = float(expr[0][4:])
  sub_power = float(expr[1][0][4:])
  new_power = sub_power * op_power
  if new_power == 1.:
    return expr[1][1]
  else:
    return ('#Pow%.4e' % new_power, expr[1][1])


@_register_simplify_fn
def log_pow_simplify(expr):
  if expr[0] == '#Log' and expr[1][0][:4] == '#Pow':
    return ('#Mul', (expr[1][0][4:],), ('#Log', expr[1][1]))


@_register_simplify_fn
def log_mul_simplify(expr):
  if expr[0] == '#Log' and expr[1][0] == '#Mul':
    return ('#Add',) + tuple(('#Log', i) for i in expr[1][1:])


@_register_simplify_fn
def pow_mul_simplify(expr):
  if expr[0][:4] == '#Pow' and expr[1][0] == '#Mul':
    return ('#Mul',) + tuple(((expr[0], i) for i in expr[1][1:]))


@_register_simplify_fn
def mul_add_simplify(expr):
  '''Turns Mul(Add(.), .) into Add(Mul(.), Mul(.),...)'''
  if expr[0] != '#Mul':
    return None
  for i in xrange(1, len(expr)):
    if expr[i][0] == '#Add':
      other_args = expr[1:i] + expr[i+1:]
      return ('#Add',) + tuple((('#Mul',) + other_args + (j,) for j in expr[i][1:]))


def commutative_simplify(expr, op_name):
  if expr[0] != op_name:
    return None
  new_args = []
  did_something = False
  for i in expr[1:]:
    if i[0] == op_name:
      new_args += i[1:]
      did_something = True
    else:
      new_args.append(i)
  if did_something:
    return (op_name,) + tuple(new_args)


@_register_simplify_fn
def add_add_simplify(expr):
  return commutative_simplify(expr, '#Add')


@_register_simplify_fn
def mul_mul_simplify(expr):
  return commutative_simplify(expr, '#Mul')


def identity_simplify(expr, op_name, identity_val):
  if expr[0] != op_name:
    return None
  if len(expr) == 2:
    return expr[1]
  new_args = []
  did_something = False
  for i in expr[1:]:
    if i[0] != identity_val:
      new_args.append(i)
    else:
      did_something = True
  if did_something:
    if len(new_args) > 1:
      return (op_name,) + tuple(new_args)
    else:
      return new_args[0]


@_register_simplify_fn
def mul_one_simplify(expr):
  return identity_simplify(expr, '#Mul', 1)


@_register_simplify_fn
def add_zero_simplify(expr):
  return identity_simplify(expr, '#Add', 0)


@_register_simplify_fn
def mul_zero_simplify(expr):
  if expr[0] != '#Mul':
    return None
  for i in expr[1:]:
    if i[0] == 0:
      return (0,)


def expr_contains(expr, node_type):
  if expr[0] == node_type:
    return True
  for i in expr[1:]:
    if expr_contains(i, node_type):
      return True
  return False


@_register_simplify_fn
def add_const_simplify(expr):
  '''Prunes branches not containing any #Identity nodes.'''
  if expr[0] != '#Add':
    return None
  did_something = False
  new_args = []
  for i in xrange(1, len(expr)):
    if expr_contains(expr[i], '#x'):
      new_args.append(expr[i])
    else:
      did_something = True
  if did_something:
    return ('#Add',) + tuple(new_args)
