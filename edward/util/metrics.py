from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from functools import wraps

import tensorflow as tf


def with_binary_averaging(metric):
  """
  Inspired by scikit-learn's _average_binary_score function: http://bit.ly/2yhcABp.

  #TODO: Complete docstring (once we're happy with this function).
  """
  average_options = (None, 'micro', 'macro')
  @wraps(metric)
  def with_binary_averaging(*args, **kwargs):
    y_true, y_pred = args
    average = kwargs.get('average', 'macro')
    if average not in average_options:
      raise ValueError('average has to be one of {0}'
                       ''.format(average_options))
    if average is None:
      return metric(y_true, y_pred)
    if average == 'macro':
      return tf.reduce_mean(metric(y_true, y_pred))
    if average == 'micro':
      y_true = tf.reshape(y_true, [-1])
      y_pred = tf.reshape(y_pred, [-1])
      return metric(y_true, y_pred)
  return with_binary_averaging
