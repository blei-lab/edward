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

_suff_stat_to_dist = {}
#_suff_stat_to_dist[('_log1m', 'log')] = lambda p1, p2: rvs.Beta(p2+1, p1+1)
#_suff_stat_to_dist[('#Identity', '#Log')] = lambda p1, p2: rvs.Gamma(p2+1, -p1)
_suff_stat_to_dist[(('#Pow-1.0000e+00', ('#x',)), (u'#Log', ('#x',)))] = lambda p1, p2: rvs.InverseGamma(-p2-1, -p1)
def normal_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p1)
  mu = sigmasq * p2
  return rvs.Normal(mu, tf.sqrt(sigmasq))
_suff_stat_to_dist[(('#Pow2.0000e+00', ('#x',)), ('#x',))] = normal_from_natural_params


def complete_conditional(rv, blanket):
  log_joint = 0
  for b in blanket:
    if getattr(b, "conjugate_log_prob", None) is None:
      raise NotImplementedError("conjugate_log_prob not implemented for {}".format(type(b)))
    log_joint += tf.reduce_sum(b.conjugate_log_prob())

  stop_nodes = set([i.value() for i in blanket])
  subgraph = extract_subgraph(log_joint, stop_nodes)
  s_stats = suff_stat_nodes(subgraph, rv.value(), blanket)
  s_stats = list(set(s_stats))

  s_stat_exprs = defaultdict(list)
  for i in xrange(len(s_stats)):
    expr = symbolic_suff_stat(s_stats[i], rv.value(), stop_nodes)
    expr = full_simplify(expr)
    multipliers_i, s_stats_i = extract_s_stat_multipliers(expr)
    s_stat_exprs[s_stats_i].append((s_stats[i],
                                    reconstruct_multiplier(multipliers_i)))

  s_stat_keys = s_stat_exprs.keys()
  order = np.argsort([str(i) for i in s_stat_keys])
  dist_key = tuple((s_stat_keys[i] for i in order))
  dist_constructor = _suff_stat_to_dist.get(dist_key, None)
  if dist_constructor is None:
    raise NotImplementedError('Conditional distribution has sufficient '
                              'statistics %s, but no available '
                              'exponential-family distribution has those '
                              'sufficient statistics.' % str(dist_key))

  s_stat_nodes = []
  s_stat_placeholders = []
  for s_stat_type in s_stat_exprs.values():
    for pair in s_stat_type:
      s_stat_nodes.append(pair[0])
      s_stat_placeholders.append(tf.placeholder(np.float32, shape=pair[0].get_shape()))
  swap_dict = {}
  for i in blanket:
    swap_dict[i.value()] = tf.placeholder(np.float32)
  for i, j in zip(s_stat_nodes, s_stat_placeholders):
    swap_dict[i] = j
  swap_back = {j: i for i, j in swap_dict.iteritems()}
  log_joint_copy = edward.util.copy(log_joint, swap_dict)
  all_nat_params = tf.gradients(log_joint_copy, s_stat_placeholders)

  nat_params = []
  i = 0
  for s_stat_type in s_stat_exprs.values():
    nat_params.append(0.)
    for pair in s_stat_type:
      nat_params[-1] += pair[1] * all_nat_params[i]
      i += 1
  for i in xrange(len(nat_params)):
    nat_params[i] = edward.util.copy(nat_params[i], swap_back)
  nat_params = [nat_params[i] for i in order]

#   natural_parameters = []
#   for i in order:
#     param_i = 0.
#     node_multiplier_list = s_stat_exprs[s_stat_keys[i]]
#     for j in xrange(len(node_multiplier_list)):
#       nat_param = tf.gradients(log_joint, node_multiplier_list[j][0])[0]
#       param_i += nat_param * node_multiplier_list[j][1]
#     natural_parameters.append(param_i)

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
                 'GatherNd', 'Squeeze', 'Concat', 'ExpandDims']
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
