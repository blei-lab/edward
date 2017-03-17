from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import defaultdict

import numpy as np
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models import random_variables as rvs


_suff_stat_to_dist = {}
#_suff_stat_to_dist[('_log1m', 'log')] = lambda p1, p2: rvs.Beta(p2+1, p1+1)
#_suff_stat_to_dist[('#Identity', '#Log')] = lambda p1, p2: rvs.Gamma(p2+1, -p1)
_suff_stat_to_dist[(('#Pow-1.0000e+00', (u'#Identity', ('#x',))), (u'#Log', (u'#Identity', ('#x',))))] = lambda p1, p2: rvs.InverseGamma(-p2-1, -p1)
def normal_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p1)
  mu = sigmasq * p2
  return rvs.Normal(mu, tf.sqrt(sigmasq))
_suff_stat_to_dist[(('#Pow2.0000e+00', (u'#Identity', ('#x',))), (u'#Identity', ('#x',)))] = normal_from_natural_params


def complete_conditional(rv, blanket):
  log_joint = 0
  for b in blanket:
    if getattr(b, "conjugate_log_prob", None) is None:
      raise NotImplementedError("conjugate_log_prob not implemented for {}".format(type(b)))
    log_joint += tf.reduce_sum(b.conjugate_log_prob())

  stop_nodes = set([i.value() for i in blanket])
  subgraph = extract_subgraph(log_joint, stop_nodes)
  s_stats = suff_stat_nodes(subgraph, rv.value(), blanket)

  s_stat_exprs = defaultdict(list)
  for i in xrange(len(s_stats)):
    expr = symbolic_suff_stat(s_stats[i], rv.value(), stop_nodes)
    expr = simplify(expr, _simplify_fns)
    multipliers_i, s_stats_i = extract_s_stat_multipliers(expr)
    s_stat_exprs[s_stats_i].append((s_stats[i], reconstruct_multiplier(multipliers_i)))

  s_stat_keys = s_stat_exprs.keys()
  order = np.argsort([str(i) for i in s_stat_keys])
  dist_key = tuple((s_stat_keys[i] for i in order))
  dist_constructor = _suff_stat_to_dist.get(dist_key, None)
  if dist_constructor is None:
    raise NotImplementedError('Conditional distribution has sufficient '
                              'statistics %s, but no available '
                              'exponential-family distribution has those '
                              'sufficient statistics.' % str(dist_key))

  natural_parameters = []
  for i in order:
    param_i = 0.
    node_multiplier_list = s_stat_exprs[s_stat_keys[i]]
    for j in xrange(len(node_multiplier_list)):
      nat_param = tf.gradients(log_joint, node_multiplier_list[j][0])[0]
      param_i += nat_param * node_multiplier_list[j][1]
    natural_parameters.append(param_i)

  return dist_constructor(*natural_parameters)


def extract_s_stat_multipliers(expr):
  if expr[0] != '#Mul':
    return (), expr
  s_stats = []
  multipliers = []
  for i in expr[1:]:
    if expr_contains(i, '#x'):
      multiplier, s_stat = extract_s_stat_multipliers(i)
      multipliers += multiplier
      s_stats += s_stat
    else:
      multipliers.append(i)
  return tuple(multipliers), tuple(s_stats)


def reconstruct_multiplier(multipliers):
  result = 1.
  for m in multipliers:
    result = result * reconstruct_expr(m)
  return result


def extract_subgraph(root, stop_nodes=set()):
  '''Copies the TF graph structure into something more pythonic.
  '''
  result = [root]
  for input in root.op.inputs:
    if input in stop_nodes:
      result.append((input,))
    else:
      result.append(extract_subgraph(input, stop_nodes))
  return tuple(result)


def subgraph_leaves(subgraph):
  '''Returns a list of leaf nodes from extract_subgraph().
  '''
  if len(subgraph) == 1:
    return subgraph
  else:
    result = []
    for input in subgraph[1:]:
      result += subgraph_leaves(input)
    return tuple(result)


def is_child(subgraph, node, stop_nodes):
  if len(subgraph) == 1:
    return subgraph[0] == node
  for input in subgraph[1:]:
    if input not in stop_nodes and is_child(input, node, stop_nodes):
      return True
  return False
  

_linear_types = ['Add', 'AddN', 'Sub', 'Mul', 'Neg', 'Identity', 'Sum',
                 'Assert', 'Reshape', 'Slice', 'StridedSlice', 'Gather',
                 'GatherNd', 'Squeeze', 'Concat', 'ExpandDims']
def suff_stat_nodes(subgraph, node, stop_nodes):
  if len(subgraph) == 1:
    if subgraph[0] == node:
      return (node,)
    else:
      return ()
  if subgraph[0].op.type == 'Identity' and subgraph[1][0] == node:
    return (subgraph[0],)
  if subgraph[0].op.type in _linear_types:
    result = []
    for input in subgraph[1:]:
      result += suff_stat_nodes(input, node, stop_nodes)
    return tuple(result)
  else:
    if is_child(subgraph, node, stop_nodes):
      return (subgraph[0],)
    else:
      return ()


# def suff_stat_str(node, base_node, stop_nodes):
#   if node == base_node:
#       return 'x'
#   elif node in stop_nodes:
#       return ''
#   sub_canons = [canonicalize_suff_stat(i, base_node, stop_nodes)
#                 for i in node.op.inputs]
#   return '%s(%s)' % (node.op.type, ','.join(sub_canons))


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


# TODO(mhoffman): Repeat until convergence.
def simplify(expr, simplify_fns):
  while True:
    did_something = False
    for fn in simplify_fns:
      did_something_i, expr = fn(expr)
      did_something = did_something or did_something_i
    if not did_something:
      break
  return expr


_simplify_fns = []
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


### Conjugate log prob functions
def _canonical_value(x):
  return x
#   if isinstance(x, RandomVariable):
#     return x.value()
#   else:
#     return x


def beta_log_prob(self):
  val = self
  a = _canonical_value(self.parameters['a'])
  b = _canonical_value(self.parameters['b'])
  result = ((a) - 1) * tf.log(val)
  result += ((b) - 1) * tf.log(1. - (val))
  result += -tf.lgamma(a) - tf.lgamma(b) + tf.lgamma(a + b)
  return result
rvs.Beta.conjugate_log_prob = beta_log_prob


def bernoulli_log_prob(self):
  val = self
  p = _canonical_value(self.parameters['p'])
  f_val = tf.cast(val, np.float32)
  return ((f_val) * tf.log(p) +
          (1. - (f_val)) * tf.log(1. - p))
rvs.Bernoulli.conjugate_log_prob = bernoulli_log_prob


def gamma_log_prob(self):
  val = self
  alpha = _canonical_value(self.parameters['alpha'])
  beta = _canonical_value(self.parameters['beta'])
  result = ((alpha) - 1) * tf.log(val)
  result -= (beta) * (val)
  result += -tf.lgamma(alpha) + (alpha) * tf.log(beta)
  return result
rvs.Gamma.conjugate_log_prob = gamma_log_prob


def poisson_log_prob(self):
  val = self
  lam = _canonical_value(self.parameters['lam'])
  f_val = tf.cast(val, np.float32)
  result = (f_val) * log(lam)
  result += -lam - tf.lgamma(f_val+1)
  return result
rvs.Poisson.conjugate_log_prob = poisson_log_prob


def normal_log_prob(self):
  val = self
  mu = _canonical_value(self.parameters['mu'])
  sigma = _canonical_value(self.parameters['sigma'])
  prec = tf.reciprocal(tf.square(sigma))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(mu)
                   + (val) * (mu))
  result -= tf.log(sigma) + 0.5 * np.log(2*np.pi)
  return result
rvs.Normal.conjugate_log_prob = normal_log_prob


def inverse_gamma_log_prob(self):
  val = self
  alpha = _canonical_value(self.parameters['alpha'])
  beta = _canonical_value(self.parameters['beta'])
  result = -((alpha) + 1) * tf.log(val)
  result -= (beta) * tf.reciprocal(val)
  result += -tf.lgamma(alpha) + (alpha) * tf.log(beta)
  return result
rvs.InverseGamma.conjugate_log_prob = inverse_gamma_log_prob
