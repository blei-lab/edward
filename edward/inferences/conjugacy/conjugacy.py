from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward.inferences.conjugacy.conjugate_log_probs
import numpy as np
import six
import tensorflow as tf
import time

from collections import defaultdict
from edward.inferences.conjugacy.simplify \
    import symbolic_suff_stat, full_simplify, expr_contains, reconstruct_expr
from edward.models import random_variables as rvs
from edward.util import copy, get_blanket


def mvn_diag_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p1)
  loc = sigmasq * p2
  return {'loc': loc, 'scale_diag': tf.sqrt(sigmasq)}


def normal_from_natural_params(p1, p2):
  sigmasq = 0.5 * tf.reciprocal(-p1)
  loc = sigmasq * p2
  return {'loc': loc, 'scale': tf.sqrt(sigmasq)}


_suff_stat_to_dist = defaultdict(dict)
_suff_stat_to_dist['binary'][(('#x',),)] = (
    rvs.Bernoulli, lambda p1: {'logits': p1})
_suff_stat_to_dist['01'][(('#Log', ('#One_minus', ('#x',))),
                          ('#Log', ('#x',)))] = (
    rvs.Beta, lambda p1, p2: {'concentration1': p2 + 1,
                              'concentration0': p1 + 1})
_suff_stat_to_dist['categorical'][(('#OneHot', ('#x',),),)] = (
    rvs.Categorical, lambda p1: {'logits': p1})
_suff_stat_to_dist['nonnegative'][(('#Log', ('#x',)),)] = (
    rvs.Chi2, lambda p1: {'df': 2.0 * (p1 + 1)})
_suff_stat_to_dist['simplex'][(('#Log', ('#x',)),)] = (
    rvs.Dirichlet, lambda p1: {'concentration': p1 + 1})
_suff_stat_to_dist['nonnegative'][(('#x',),)] = (
    rvs.Exponential, lambda p1: {'rate': -p1})
_suff_stat_to_dist['nonnegative'][(('#Log', ('#x',)),
                                   ('#x',))] = (
    rvs.Gamma, lambda p1, p2: {'concentration': p1 + 1, 'rate': -p2})
_suff_stat_to_dist['nonnegative'][(('#CPow-1.0000e+00', ('#x',)),
                                   ('#Log', ('#x',)))] = (
    rvs.InverseGamma, lambda p1, p2: {'concentration': -p2 - 1, 'rate': -p1})
_suff_stat_to_dist['multivariate_real'][(('#CPow2.0000e+00', ('#x',)),
                                        ('#x',))] = (
    rvs.MultivariateNormalDiag, mvn_diag_from_natural_params)
_suff_stat_to_dist['real'][(('#CPow2.0000e+00', ('#x',)),
                            ('#x',))] = (
    rvs.Normal, normal_from_natural_params)
_suff_stat_to_dist['countable'][(('#x',),)] = (
    rvs.Poisson, lambda p1: {'rate': tf.exp(p1)})


def complete_conditional(rv, cond_set=None):
  """Returns the conditional distribution ``RandomVariable``
  :math:`p(\\text{rv} | \\cdot)`.

  This function tries to infer the conditional distribution of ``rv``
  given ``cond_set``, a set of other ``RandomVariable``s in the graph. It
  will only be able to do this if

  1. :math:`p(\\text{rv} | \\text{cond_set})` is in a tractable
     exponential family, AND
  2. the truth of assumption 1 is not obscured in the TensorFlow graph.

  In other words, this function will do its best to recognize conjugate
  relationships when they exist. But it may not always be able to do the
  necessary algebra.

  Parameters
  ----------
  rv : RandomVariable
    The random variable whose conditional distribution we are interested in.
  cond_set : iterable of RandomVariable, optional
    The set of random variables we want to condition on. Default is all
    random variables in the graph. (It makes no difference if ``cond_set``
    does or does not include ``rv``.)

  Notes
  -----
  When calling ``complete_conditional()`` multiple times, one should
  usually pass an explicit ``cond_set``. Otherwise
  ``complete_conditional()`` will try to condition on the
  ``RandomVariable``s returned by previous calls to itself. This may
  result in unpredictable behavior.
  """
  if cond_set is None:
    # Default to Markov blanket, excluding conditionals. This is useful if
    # calling complete_conditional many times without passing in cond_set.
    cond_set = get_blanket(rv)
    cond_set = [i for i in cond_set if not
                ('complete_conditional' in i.unique_name and
                 'cond_dist' in i.unique_name)]

  cond_set = set([rv] + list(cond_set))
  with tf.name_scope('complete_conditional_%s' % rv.unique_name) as scope:
    # log_joint holds all the information we need to get a conditional.
    log_joint = get_log_joint(cond_set)

    # Pull out the nodes that are nonlinear functions of rv into s_stats.
    stop_nodes = set([i.value() for i in cond_set])
    subgraph = extract_subgraph(log_joint, stop_nodes)
    s_stats = suff_stat_nodes(subgraph, rv.value(), cond_set)
    s_stats = list(set(s_stats))

    # Simplify those nodes, and put any new linear terms into multipliers_i.
    s_stat_exprs = defaultdict(list)
    for s_stat in s_stats:
      expr = symbolic_suff_stat(s_stat, rv.value(), stop_nodes)
      expr = full_simplify(expr)
      multipliers_i, s_stats_i = extract_s_stat_multipliers(expr)
      s_stat_exprs[s_stats_i].append(
          (s_stat, reconstruct_multiplier(multipliers_i)))

    # Sort out the sufficient statistics to identify this conditional's family.
    s_stat_keys = list(six.iterkeys(s_stat_exprs))
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
    # w.r.t. those placeholders to get natural parameters. The original
    # nodes involving the sufficient statistic nodes are swapped for new
    # nodes that depend linearly on the sufficient statistic placeholders.
    s_stat_placeholders = []
    swap_dict = {}
    swap_back = {}
    for s_stat_expr in six.itervalues(s_stat_exprs):
      s_stat_placeholder = tf.placeholder(tf.float32,
                                          s_stat_expr[0][0].get_shape())
      swap_back[s_stat_placeholder] = tf.cast(rv.value(), tf.float32)
      s_stat_placeholders.append(s_stat_placeholder)
      for s_stat_node, multiplier in s_stat_expr:
        fake_node = s_stat_placeholder * multiplier
        swap_dict[s_stat_node] = fake_node
        swap_back[fake_node] = s_stat_node

    for i in cond_set:
      if i != rv:
        val = i.value()
        val_placeholder = tf.placeholder(val.dtype)
        swap_dict[val] = val_placeholder
        swap_back[val_placeholder] = val
        swap_back[val] = val  # prevent random variable nodes from being copied

    scope_name = scope + str(time.time())  # ensure unique scope when copying
    log_joint_copy = copy(log_joint, swap_dict, scope=scope_name + 'swap')
    nat_params = tf.gradients(log_joint_copy, s_stat_placeholders)

    # Remove any dependencies on those old placeholders.
    nat_params = [copy(nat_param, swap_back, scope=scope_name + 'swapback')
                  for nat_param in nat_params]
    nat_params = [nat_params[i] for i in order]

    return dist_constructor(name='cond_dist', **constructor_params(*nat_params))


def get_log_joint(cond_set):
  g = tf.get_default_graph()
  cond_set_names = [i.unique_name[:-1] for i in cond_set]
  cond_set_names.sort()
  cond_set_name = 'log_joint_of_' + '_'.join(cond_set_names)
  with tf.name_scope("conjugate_log_joint/") as scope:
    try:
      # Use log joint tensor if already built in graph.
      return g.get_tensor_by_name(scope + cond_set_name + ':0')
    except:
      pass

    terms = []
    for b in cond_set:
      name = b.unique_name.replace(':', '_') + '_conjugate_log_prob'
      try:
        # Use log prob tensor if already built in graph.
        conjugate_log_prob = g.get_tensor_by_name(scope + name + ':0')
      except:
        if getattr(b, "conjugate_log_prob", None) is None:
          raise NotImplementedError("conjugate_log_prob not implemented for"
                                    " {}".format(type(b)))
        conjugate_log_prob = tf.reduce_sum(b.conjugate_log_prob(), name=name)

      terms.append(conjugate_log_prob)

    result = tf.add_n(terms, name=cond_set_name)
    return result


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
    result *= reconstruct_expr(m)
  return result


def extract_subgraph(root, stop_nodes=set()):
  """Copies the TF graph structure into something more pythonic.
  """
  result = [root]
  for input in root.op.inputs:
    if input in stop_nodes:
      result.append((input,))
    else:
      result.append(extract_subgraph(input, stop_nodes))
  return tuple(result)


def subgraph_leaves(subgraph):
  """Returns a list of leaf nodes from extract_subgraph().
  """
  if len(subgraph) == 1:
    return subgraph
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
  """Finds nonlinear nodes depending on `node`.
  """
  if subgraph[0] == node:
    return (node,)
  elif len(subgraph) == 1:
    return ()
  op_type = str(subgraph[0].op.type)
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
