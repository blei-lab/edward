from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _mul_n(x):
  if len(x) == 2:
    return tf.multiply(x[0], x[1])
  else:
    return tf.multiply(x[0], _mul_n(x[1:]))


_extractable_nodes = {
    'Add': tf.add_n,
    'Sub': tf.subtract,
    'Mul': _mul_n,
    'Log': tf.log,
    'Exp': tf.exp,
    'Pow': tf.pow,
    'Square': tf.square,
    'Reciprocal': tf.reciprocal,
    'Sqrt': tf.sqrt,
    'Identity': lambda x: x,
    'One_minus': lambda x: 1 - x,
    # Makes some assumptions.
    'OneHot': lambda x: tf.one_hot(x, tf.reduce_max(x) + 1, dtype=tf.float32)
}


def symbolic_suff_stat(node, base_node, stop_nodes):
  """Extracts a symbolic representation of the graph rooted at `node`.
  """
  if node == base_node:
    return ('#x',)
  elif node in stop_nodes:
    return (node,)

  if node.op.type in _extractable_nodes:
    result = ['#%s' % str(node.op.type)]
  else:
    result = [node]

  result += [symbolic_suff_stat(i, base_node, stop_nodes)
             for i in node.op.inputs]
  return tuple(result)


def is_number(x):
  if isinstance(x, tf.Tensor):
    return True
  try:
    float(x)
    return True
  except:
    return False


def reconstruct_expr(expr):
  if is_number(expr[0]):
    return expr[0]
  if expr[0] == '#x':
    raise ValueError('#x cannot appear in expr to be reconstructed.')
  args = [reconstruct_expr(i) for i in expr[1:]]
  if str(expr[0])[:5] == '#CPow':
    return tf.pow(args[0], np.float32(expr[0][5:]))
  if expr[0][0] == '#':
    tf_fn = _extractable_nodes.get(expr[0][1:], None)
    assert(tf_fn is not None)
    return tf_fn(*args)
  assert(False)


_simplify_fns = []


def full_simplify(expr, simplify_fns=_simplify_fns):
  while True:
    did_something = False
    for fn in simplify_fns:
      did_something_i, expr = fn(expr)
      did_something = did_something or did_something_i
    if not did_something:
      break
  return expr


def _register_simplify_fn(fn):
  """Wraps and registers simplification functions.

  A simplification function takes as input an expression and possible
  some other args/kwargs, and returns either None (if it did not find
  anything to do at this node) or a simplified version of the graph
  below this node.

  The wrapped function will repeatedly apply this simplification
  function to all nodes of the graph until it stops doing anything.
  """
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


@_register_simplify_fn
def identity_op_simplify(expr):
  if expr[0] == '#Identity':
    return expr[1]


_power_ops = {
    '#Reciprocal': -1.,
    '#Square': 2.,
    '#Sqrt': 0.5,
}


@_register_simplify_fn
def power_op_simplify(expr):
  op_power = _power_ops.get(expr[0], None)
  if op_power:
    return ('#CPow%.4e' % op_power,) + expr[1:]


@_register_simplify_fn
def pow_simplify(expr):
  if str(expr[0])[:5] != '#CPow':
    return None
  if str(expr[1][0])[:5] != '#CPow':
    return None

  op_power = float(expr[0][5:])
  sub_power = float(expr[1][0][5:])
  new_power = sub_power * op_power
  if new_power == 1.:
    return expr[1][1]
  else:
    return ('#CPow%.4e' % new_power, expr[1][1])


@_register_simplify_fn
def log_pow_simplify(expr):
  if expr[0] == '#Log' and str(expr[1][0])[:5] == '#CPow':
    return ('#Mul', (float(expr[1][0][5:]),), ('#Log', expr[1][1]))
  if expr[0] == '#Log' and expr[1][0] == '#Pow':
    return ('#Mul', expr[1][2], ('#Log', expr[1][1]))


@_register_simplify_fn
def log_mul_simplify(expr):
  if expr[0] == '#Log' and expr[1][0] == '#Mul':
    return ('#Add',) + tuple(('#Log', i) for i in expr[1][1:])


@_register_simplify_fn
def pow_mul_simplify(expr):
  if str(expr[0])[:5] == '#CPow' and expr[1][0] == '#Mul':
    return ('#Mul',) + tuple(((expr[0], i) for i in expr[1][1:]))
  if expr[0] == '#Pow' and expr[1][0] == '#Mul':
    return ('#Mul',) + tuple((('#Pow', i, expr[2]) for i in expr[1][1:]))


@_register_simplify_fn
def mul_add_simplify(expr):
  """Turns Mul(Add(.), .) into Add(Mul(.), Mul(.),...)"""
  if expr[0] != '#Mul':
    return None
  for i in range(1, len(expr)):
    if expr[i][0] == '#Add':
      other_args = expr[1:i] + expr[i + 1:]
      return ('#Add',) + tuple((('#Mul',) + other_args + (j,)
                                for j in expr[i][1:]))


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


@_register_simplify_fn
def square_add_simplify(expr):
  if not (expr[0] == '#CPow2.0000e+00' and expr[1][0] == '#Add'):
    return None
  terms = []
  for i in range(1, len(expr[1])):
    terms.append(('#CPow2.0000e+00', expr[1][i]))
    for j in range(i + 1, len(expr[1])):
      terms.append(('#Mul', (2.0,), expr[1][i], expr[1][j]))
  return ('#Add',) + tuple(terms)


def expr_contains(expr, node_type):
  if expr[0] == node_type:
    return True
  for i in expr[1:]:
    if expr_contains(i, node_type):
      return True
  return False


@_register_simplify_fn
def add_const_simplify(expr):
  """Prunes branches not containing any #x nodes."""
  if expr[0] != '#Add':
    return None
  did_something = False
  new_args = []
  for i in range(1, len(expr)):
    if expr_contains(expr[i], '#x'):
      new_args.append(expr[i])
    else:
      did_something = True
  if did_something:
    return ('#Add',) + tuple(new_args)


@_register_simplify_fn
def one_m_simplify(expr):
  """Replaces ("#Sub", (<wrapped constant 1>,), (.)) with ("#One_minus", .)."""
  if expr[0] != '#Sub' or not isinstance(expr[1][0], tf.Tensor):
    return None
  value = tf.contrib.util.constant_value(expr[1][0].op.outputs[0])
  if value == 1.0:
    return ('#One_minus', expr[2])


@_register_simplify_fn
def cast_simplify(expr):
  """Replaces (<wrapped cast>, (.)) with (.)."""
  if isinstance(expr[0], tf.Tensor) and expr[0].op.type == 'Cast':
    return expr[1]


@_register_simplify_fn
def onehot_simplify(expr):
  """Gets rid of extraneous args to OneHot."""
  if expr[0] == '#OneHot' and len(expr) > 2:
    return expr[:2]
