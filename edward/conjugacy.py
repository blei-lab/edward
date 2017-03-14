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
_suff_stat_to_dist['_log1m|log'] = lambda p1, p2: rvs.Beta(p2+1, p1+1)

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
  s_stat_names = [s_stat_names[i] for i in order]
  s_stat_nodes = [s_stats[i][1] for i in order]

  s_stat_names = '|'.join(s_stat_names)
  # TODO(mhoffman): Make a nicer exception.
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


def log(x):
  return sufficient_statistic(tf.log, x)


def square(x):
  return sufficient_statistic(tf.square, x)


def reciprocal(x):
  return sufficient_statistic(tf.reciprocal, x)


def _log1m(x):
  return tf.log1p(-x)


def log1m(x):
  return sufficient_statistic(_log1m, x)


def _canonical_value(x):
  if isinstance(x, RandomVariable):
    return x.value()
  else:
    return x


def beta_log_prob(self):
  val = self.value()
  a = _canonical_value(self.parameters['a'])
  b = _canonical_value(self.parameters['b'])
  result = (a - 1) * log(val) + (b - 1) * log1m(val)
  result += -tf.lgamma(a) - tf.lgamma(b) + tf.lgamma(a + b)
  return result
rvs.Beta.conjugate_log_prob = beta_log_prob
print(rvs.Beta.conjugate_log_prob)


def bernoulli_log_prob(self):
  val = self.value()
  p = _canonical_value(self.parameters['p'])
  return (tf.cast(val, np.float32) * log(p)
          + (1 - tf.cast(val, np.float32)) * log1m(p))
rvs.Bernoulli.conjugate_log_prob = bernoulli_log_prob


#### CRUFT

def print_tree(op, depth=0, stop_nodes=None, stop_types=None):
  if stop_nodes is None: stop_nodes = set()
  if stop_types is None: stop_types = set()
  print(''.join(['-'] * depth), '%s...%s' % (op.type, op.name))
  if (op not in stop_nodes) and (op.type not in stop_types):
    for i in op.inputs:
      print_tree(i.op, depth+1, stop_nodes=stop_nodes, stop_types=stop_types)
