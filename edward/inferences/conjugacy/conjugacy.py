from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import defaultdict
from pprint import pprint
from copy import copy

import numpy as np
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models import random_variables as rvs

import edward.inferences.conjugacy.conjugate_log_probs
from edward.inferences.conjugacy.simplify \
    import symbolic_suff_stat, full_simplify, expr_contains, reconstruct_expr

copy_op_to_graph = tf.contrib.copy_graph.copy_op_to_graph

# TODO(mhoffman): Support for slicing, tf.gather, etc.


def normal_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p1)
  mu = sigmasq * p2
  return {'mu': mu, 'sigma': tf.sqrt(sigmasq)}


rvs.Bernoulli.support = 'binary'
rvs.Categorical.support = 'onehot'
rvs.Beta.support = '01'
rvs.Dirichlet.support = 'simplex'
rvs.Gamma.support = 'nonnegative'
rvs.InverseGamma.support = 'nonnegative'
rvs.Normal.support = 'real'


_suff_stat_to_dist = defaultdict(dict)
_suff_stat_to_dist['binary'][(('#x',),)] = (
    rvs.Bernoulli, lambda p1: {'p': tf.sigmoid(p1)})
_suff_stat_to_dist['onehot'][(('#OneHot', ('#x',),),)] = (
    rvs.Categorical, lambda p1: {'p': tf.nn.softmax(p1)})
_suff_stat_to_dist['01'][((u'#Log', ('#One_minus', ('#x',))),
                          (u'#Log', ('#x',)))] = (
    rvs.Beta, lambda p1, p2: {'a': p2 + 1, 'b': p1 + 1})
_suff_stat_to_dist['simplex'][((u'#Log', ('#x',)),)] = (
    rvs.Dirichlet, lambda p1: {'alpha': p1 + 1})
_suff_stat_to_dist['nonnegative'][(('#x',),
                                   (u'#Log', ('#x',)))] = (
    rvs.Gamma, lambda p1, p2: {'alpha': p2 + 1, 'beta': -p1})
_suff_stat_to_dist['nonnegative'][(('#CPow-1.0000e+00', ('#x',)),
                                   (u'#Log', ('#x',)))] = (
    rvs.InverseGamma, lambda p1, p2: {'alpha': -p2 - 1, 'beta': -p1})
_suff_stat_to_dist['real'][(('#CPow2.0000e+00', ('#x',)),
                            ('#x',))] = (
    rvs.Normal, normal_from_natural_params)


def _log_joint_name(blanket):
  return '_log_joint_of_' + ('&'.join([i.name[:-1] for i in blanket])) + '_'


def get_log_joint(blanket):
  g = tf.get_default_graph()
  blanket_name = _log_joint_name(blanket)
  c = g.get_collection(blanket_name)
  if len(c):
    return c[0]

  with tf.name_scope('conjugate_log_joint') as scope:
    terms = []
    for b in blanket:
      if getattr(b, "conjugate_log_prob", None) is None:
        raise NotImplementedError("conjugate_log_prob not implemented for"
                                  " {}".format(type(b)))
      terms.append(tf.reduce_sum(b.conjugate_log_prob()))
    result = tf.add_n(terms, name=scope)
    g.add_to_collection(blanket_name, result)
    return result


def complete_conditional(rv, blanket, log_joint=None):
  with tf.name_scope('complete_conditional_%s' % rv.name) as scope:
    # log_joint holds all the information we need to get a conditional.
    extended_blanket = copy(blanket)
    if log_joint is None:
      log_joint = get_log_joint(extended_blanket)
    else:
      log_joint = log_joint(extended_blanket)

    # Pull out the nodes that are nonlinear functions of rv into s_stats.
    stop_nodes = set([i.value() for i in blanket])
    subgraph = extract_subgraph(log_joint, stop_nodes)
    s_stats = suff_stat_nodes(subgraph, rv.value(), blanket)
    s_stats = list(set(s_stats))

    # Simplify those nodes, and extract any new linear terms into multipliers_i.
    s_stat_exprs = defaultdict(list)
    for i in range(len(s_stats)):
      expr = symbolic_suff_stat(s_stats[i], rv.value(), stop_nodes)
      expr = full_simplify(expr)
      multipliers_i, s_stats_i = extract_s_stat_multipliers(expr)
      s_stat_exprs[s_stats_i].append((s_stats[i],
                                      reconstruct_multiplier(multipliers_i)))

    # Sort out the sufficient statistics to identify this conditional's family.
    s_stat_keys = s_stat_exprs.keys()
    order = np.argsort([str(i) for i in s_stat_keys])
    dist_key = tuple((s_stat_keys[i] for i in order))
    dist_constructor, constructor_params = (
        _suff_stat_to_dist[rv.support].get(dist_key, (None, None)))
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
    swap_dict = {}
    swap_back = {}
    for s_stat in s_stat_exprs.keys():
      s_stat_shape = s_stat_exprs[s_stat][0][0].get_shape()
      s_stat_placeholder = tf.placeholder(np.float32, s_stat_shape)
      swap_back[s_stat_placeholder] = tf.cast(rv.value(), np.float32)
      s_stat_placeholders.append(s_stat_placeholder)
      for s_stat_node, multiplier in s_stat_exprs[s_stat]:
        fake_node = s_stat_placeholder * multiplier
        s_stat_nodes.append(s_stat_node)
        s_stat_replacements.append(fake_node)
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

    log_joint_copy = edward.util.copy(log_joint, swap_dict,
                                      scope=scope + 'swap')
    nat_params = tf.gradients(log_joint_copy, s_stat_placeholders)

    # Removes any dependencies on those old placeholders.
    for i in range(len(nat_params)):
      nat_params[i] = edward.util.copy(nat_params[i], swap_back,
                                       scope=scope + 'swapback')
    nat_params = [nat_params[i] for i in order]

    return dist_constructor(name='cond_dist', **constructor_params(*nat_params))


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
_n_important_args = {'Sum': 1}


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
  op_type = subgraph[0].op.type
  if op_type in _linear_types:
    result = []
    stop_index = _n_important_args.get(op_type, None)
    stop_index = stop_index + 1 if stop_index is not None else None
    for input in subgraph[1:stop_index]:
      result += suff_stat_nodes(input, node, stop_nodes)
    return tuple(result)
  else:
    if is_child(subgraph, node, stop_nodes):
      return (subgraph[0],)
    else:
      return ()
