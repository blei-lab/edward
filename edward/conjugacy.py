from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import random_variables as rv


_suff_stat_registry = {}
_conj_log_prob_registry = {}


def complete_conditional(rv, blanket):
  log_joint = 0
  for b in blanket:
    log_joint += tf.reduce_sum(_conj_log_prob_registry[type(b)](b))
  return log_joint


def sufficient_statistic(f, x):
  """Returns and caches (if necessary) f(x).

  If it's been called before, returns the cached version, so that
  there is only one canonical f(x) node.
  """
  g = tf.get_default_graph()
  key = (g, f, x)
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


def log1m(x):
  return sufficient_statistic(lambda y: tf.log1p(-y), x)


def beta_log_prob(x):
  result = (x.a - 1) * log(x) + (x.b - 1) * log1m(x)
  result += -tf.lgamma(x.a) - tf.lgamma(x.b) + tf.lgamma(x.a + x.b)
  return result
# def beta_log_prob(x, a, b):
#   result = (a - 1) * log(x) + (b - 1) * log1m(x)
#   result += -tf.lgamma(a) - tf.lgamma(b) + tf.lgamma(a + b)
_conj_log_prob_registry[rv.Beta] = beta_log_prob


def bernoulli_log_prob(x):
  return (tf.cast(x, np.float32) * log(x.p)
          + (1 - tf.cast(x, np.float32)) * log1m(x.p))
# def bernoulli_log_prob(x, p):
#   return x * log(p) + (1 - x) * log1m(p)
_conj_log_prob_registry[rv.Bernoulli] = bernoulli_log_prob
