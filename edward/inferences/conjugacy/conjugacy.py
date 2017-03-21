from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import defaultdict
from pprint import pprint

import numpy as np
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models import random_variables as rvs

import edward.inferences.conjugacy.conjugate_log_probs
from edward.inferences.conjugacy.simplify import symbolic_suff_stat, full_simplify, expr_contains, reconstruct_expr

#TODO(mhoffman): Support for slicing, tf.gather, etc.

rvs.Bernoulli.support = 'binary'
rvs.Categorical.support = 'onehot'
rvs.Beta.support = '01'
rvs.Dirichlet.support = 'simplex'
rvs.Gamma.support = 'nonnegative'
rvs.InverseGamma.support = 'nonnegative'
rvs.Normal.support = 'real'

_suff_stat_to_dist = defaultdict(dict)
_suff_stat_to_dist['binary'][(('#x',),)] = lambda p1: rvs.Bernoulli(p=tf.sigmoid(p1))
_suff_stat_to_dist['01'][((u'#Log', ('#One_minus', ('#x',))), (u'#Log', ('#x',)))] = lambda p1, p2: rvs.Beta(p2+1, p1+1)
_suff_stat_to_dist['simplex'][((u'#Log', ('#x',)),)] = lambda p1: rvs.Dirichlet(p1+1)
_suff_stat_to_dist['nonnegative'][(('#x',), (u'#Log', ('#x',)))] = lambda p1, p2: rvs.Gamma(p2+1, -p1)
_suff_stat_to_dist['nonnegative'][(('#Pow-1.0000e+00', ('#x',)), (u'#Log', ('#x',)))] = lambda p1, p2: rvs.InverseGamma(-p2-1, -p1)
def normal_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p1)
  mu = sigmasq * p2
  return rvs.Normal(mu, tf.sqrt(sigmasq))
_suff_stat_to_dist['real'][(('#Pow2.0000e+00', ('#x',)), ('#x',))] = normal_from_natural_params


def complete_conditional(rv, blanket, log_joint=None):
  # log_joint holds all the information we need to get a conditional.
  if log_joint is None:
    log_joint = 0
    for b in blanket:
      if getattr(b, "conjugate_log_prob", None) is None:
        raise NotImplementedError("conjugate_log_prob not implemented for {}".format(type(b)))
      log_joint += tf.reduce_sum(b.conjugate_log_prob())

  # Pull out the nodes that are nonlinear functions of rv into s_stats.
  stop_nodes = set([i.value() for i in blanket])
  subgraph = extract_subgraph(log_joint, stop_nodes)
  s_stats = suff_stat_nodes(subgraph, rv.value(), blanket)
  s_stats = list(set(s_stats))

  # Simplify those nodes, and extract any new linear terms into multipliers_i.
  s_stat_exprs = defaultdict(list)
  for i in xrange(len(s_stats)):
    expr = symbolic_suff_stat(s_stats[i], rv.value(), stop_nodes)
    expr = full_simplify(expr)
    multipliers_i, s_stats_i = extract_s_stat_multipliers(expr)
    s_stat_exprs[s_stats_i].append((s_stats[i],
                                    reconstruct_multiplier(multipliers_i)))

  # Sort out the sufficient statistics to identify this conditional's family.
  s_stat_keys = s_stat_exprs.keys()
  order = np.argsort([str(i) for i in s_stat_keys])
  dist_key = tuple((s_stat_keys[i] for i in order))
  dist_constructor = _suff_stat_to_dist[rv.support].get(dist_key, None)
  if dist_constructor is None:
    raise NotImplementedError('Conditional distribution has sufficient '
                              'statistics %s, but no available '
                              'exponential-family distribution has those '
                              'sufficient statistics.' % str(dist_key))

  # Swap sufficient statistics for placeholders, then take gradients
  # w.r.t.  those placeholders to get natural parameters. The original
  # nodes involving the sufficient statistic nodes are swapped for new
  # nodes that depend linearly on the sufficient statistic placeholders.
  s_stat_nodes = []
  s_stat_replacements = []
  s_stat_placeholders = []
  for s_stat in s_stat_exprs.keys():
    # TODO(mhoffman): This shape assumption won't work for MVNs or Wisharts.
    s_stat_placeholder = tf.placeholder(np.float32,
                                        shape=rv.value().get_shape())
    s_stat_placeholders.append(s_stat_placeholder)
    for s_stat_node, multiplier in s_stat_exprs[s_stat]:
      fake_node = s_stat_placeholder * multiplier
      s_stat_nodes.append(s_stat_node)
      s_stat_replacements.append(fake_node)
  swap_dict = {}
  swap_back = {}
  for i in blanket:
    if i == rv:
      continue
    val = i.value()
    swap_dict[val] = tf.placeholder(val.dtype)
    swap_back[swap_dict[val]] = val
    # This prevents random variable nodes from being copied.
    swap_back[val] = val
  for i, j in zip(s_stat_nodes, s_stat_replacements):
    swap_dict[i] = j
    swap_back[j] = i

  log_joint_copy = edward.util.copy(log_joint, swap_dict)
  nat_params = tf.gradients(log_joint_copy, s_stat_placeholders)

  # Removes any dependencies on those old placeholders.
  for i in xrange(len(nat_params)):
    nat_params[i] = edward.util.copy(nat_params[i], swap_back, scope='copyback')
  nat_params = [nat_params[i] for i in order]

  return dist_constructor(*nat_params)


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
                 'GatherNd', 'Squeeze', 'Concat', 'ExpandDims', 'OneHot']
def suff_stat_nodes(subgraph, node, stop_nodes):
  '''Finds nonlinear nodes depending on `node`.
  '''
  if len(subgraph) == 1:
    if subgraph[0] == node:
      return (node,)
    else:
      return ()
  if subgraph[0] == node:
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
