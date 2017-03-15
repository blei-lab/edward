from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models import random_variables as rvs


# TODO(mhoffman): _suff_stat_registry should probably be a tf collection
_suff_stat_registry = {}
_suff_stat_to_dist = {}

# TODO(mhoffman): Support (discrete/continuous mostly) also matters.
_suff_stat_to_dist[('_log1m', 'log')] = lambda p1, p2: rvs.Beta(p2+1, p1+1)
_suff_stat_to_dist[('identity', 'log')] = lambda p1, p2: rvs.Gamma(p2+1, -p1)
def normal_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p2)
  mu = sigmasq * p1
  return rvs.Normal(mu, tf.sqrt(sigmasq))
_suff_stat_to_dist[('identity', 'square')] = normal_from_natural_params

def complete_conditional(rv, blanket):
  log_joint = 0
  for b in blanket:
    if getattr(b, "conjugate_log_prob", None) is None:
      raise NotImplementedError("conjugate_log_prob not implemented for {}".format(type(b)))
    log_joint += tf.reduce_sum(b.conjugate_log_prob())

  s_stats = []
  for i, j in _suff_stat_registry.iteritems():
    if i[2] == rv.value():
      s_stats.append((i[1], j))
  s_stat_names = [i[0] for i in s_stats]
  order = np.argsort(s_stat_names)
  s_stat_names = tuple(s_stat_names[i] for i in order)
  s_stat_nodes = [s_stats[i][1] for i in order]

  if s_stat_names not in _suff_stat_to_dist:
    raise NotImplementedError(
      "No available exponential-family distribution with sufficient statistics "
      + ', '.join(s_stat_names))
  assert(s_stat_names in _suff_stat_to_dist)

  n_params = tf.gradients(log_joint, s_stat_nodes)

  return _suff_stat_to_dist[s_stat_names](*n_params)


def de_identity(node):
  '''
  Gets rid of Identity nodes.
  TODO: Relying on this might screw up device placement.
  '''
  while (node.type == 'Identity'):
    node = list(node.inputs)[0].op
  return node


def _fn_name(f):
  return re.search(r'<function (.+) at 0x.*>', str(f)).group(1)


def sufficient_statistic(f, x):
  """Returns and caches (if necessary) f(x).

  If it's been called before, returns the cached version, so that
  there is only one canonical f(x) node.
  """
  g = tf.get_default_graph()
  key = (g, _fn_name(f), x)
  if key in _suff_stat_registry:
    return _suff_stat_registry[key]
  else:
    result = f(x)
    _suff_stat_registry[key] = result
    return result


def identity(x):
  # This function registers x as a sufficient statistic.
  # If x is a sum or product, its parents are registered instead.
  return _wrap_leaves(x, {'Add': tf.add, 'Mul': tf.multiply},
                      lambda y: sufficient_statistic(tf.identity, y))


def log(x):
  x_op = getattr(x, 'op', None)
  if x_op is None or x_op.type != 'Mul':
    return sufficient_statistic(tf.log, x)

  # Rewrite log(x * y) as log(x) + log(y)
  multiplicands = _unpack_mul(x_op)
  log_multiplicands = [log(i) for i in multiplicands]
  return tf.add_n(log_multiplicands)


def square(x):
  return sufficient_statistic(tf.square, x)


def reciprocal(x):
  return sufficient_statistic(tf.reciprocal, x)


def _log1m(x):
  return tf.log1p(-x)


def log1m(x):
  return sufficient_statistic(_log1m, x)


# TODO(mhoffman): Automate the process of adding these s.stat functions.
def lgamma(x):
  return sufficient_statistic(tf.lgamma, x)


def square(x):
  return sufficient_statistic(tf.square, x)


def _squared_reciprocal(x):
  return tf.reciprocal(tf.square(x))


def squared_reciprocal(x):
  return sufficient_statistic(_squared_reciprocal, x)


def _canonical_value(x):
  if isinstance(x, RandomVariable):
    return x.value()
  else:
    return x


def _unpack_op(x_op, types_to_unpack):
  results = []
  for parent in x_op.inputs:
    if parent.op.type in types_to_unpack:
      results += _unpack_op(parent.op, types_to_unpack)
    else:
      results += [parent]
  return results


def _unpack_mul(x_op):
  return _unpack_op(x_op, set(['Mul']))


def _wrap_leaves(x, non_leaf_types, wrapper):
  '''Walks up from x_op and wraps any nodes not in a whitelist with wrapper().

  Args:
    x: The Tensor whose parents we want to wrap up.
    non_leaf_types: A dict mapping from names (e.g., "Add") to functions
      (e.g., tf.add).
    wrapper: The function to wrap leaf nodes with.
  '''
  x_op = getattr(x, 'op', None)
  if x_op is None or x_op.type not in non_leaf_types:
    return wrapper(x)
  else:
    inputs = []
    for parent in x_op.inputs:
      inputs.append(_wrap_leaves(parent, non_leaf_types, wrapper))
    return non_leaf_types[x_op.type](*inputs)


def _unpack_mul_add(x_op):
  return _unpack_op(x_op, set(['Mul', 'Add']))


def beta_log_prob(self):
  val = self.value()
  a = _canonical_value(self.parameters['a'])
  b = _canonical_value(self.parameters['b'])
  result = (identity(a) - 1) * log(val)
  result += (identity(b) - 1) * log1m(val)
  result += -lgamma(a) - lgamma(b) + lgamma(a + b)
  return result
rvs.Beta.conjugate_log_prob = beta_log_prob


def bernoulli_log_prob(self):
  val = self.value()
  p = _canonical_value(self.parameters['p'])
  return (tf.cast(val, np.float32) * log(p)
          + (1 - tf.cast(val, np.float32)) * log1m(p))
rvs.Bernoulli.conjugate_log_prob = bernoulli_log_prob


def gamma_log_prob(self):
  val = self.value()
  alpha = _canonical_value(self.parameters['alpha'])
  beta = _canonical_value(self.parameters['beta'])
  result = (identity(alpha) - 1) * log(val) - identity(beta) * identity(val)
  result += -lgamma(alpha) + identity(alpha) * log(beta)
  return result
rvs.Gamma.conjugate_log_prob = gamma_log_prob


def poisson_log_prob(self):
  val = self.value()
  lam = _canonical_value(self.parameters['lam'])
  result = identity(tf.cast(val, np.float32)) * log(lam)
  result += - identity(lam) - tf.lgamma(val+1)
  return result
rvs.Poisson.conjugate_log_prob = poisson_log_prob


def normal_log_prob(self):
  val = self.value()
  mu = _canonical_value(self.parameters['mu'])
  sigma = _canonical_value(self.parameters['sigma'])
  prec = squared_reciprocal(sigma)
  result = prec * (-0.5 * square(val) - 0.5 * square(mu)
                   + identity(val) * identity(mu))
  result -= log(sigma) + 0.5 * np.log(2*np.pi)
  return result
rvs.Normal.conjugate_log_prob = normal_log_prob


#### CRUFT

def print_tree(op, depth=0, stop_nodes=None, stop_types=None):
  if stop_nodes is None: stop_nodes = set()
  if stop_types is None: stop_types = set()
  print(''.join(['-'] * depth), '%s...%s' % (op.type, op.name))
  if (op not in stop_nodes) and (op.type not in stop_types):
    for i in op.inputs:
      print_tree(i.op, depth+1, stop_nodes=stop_nodes, stop_types=stop_types)
