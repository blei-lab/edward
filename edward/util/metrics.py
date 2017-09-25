from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from functools import wraps

import tensorflow as tf


def with_binary_averaging(metric):
  """
  Inspired by scikit-learn's _average_binary_score function:
  http://bit.ly/2yhcABp.

  #TODO: Complete docstring (once we're happy with this function).
  """
  AVERAGE_OPTIONS = (None, 'micro', 'macro')

  @wraps(metric)
  def with_binary_averaging(*args, **kwargs):
    y_true, y_pred = args
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) < 2 and len(y_pred.shape) < 2:
      y_true = tf.expand_dims(y_true, 0)
      y_pred = tf.expand_dims(y_pred, 0)

    average = kwargs.get('average', 'macro')
    if average not in AVERAGE_OPTIONS:
      raise ValueError('average has to be one of {0}'
                       ''.format(average_options))
    if average is None:
      return metric(y_true, y_pred)
    if average == 'macro':
      return tf.reduce_mean(metric(y_true, y_pred))
    if average == 'micro':
      y_true = tf.reshape(y_true, [1, -1])
      y_pred = tf.reshape(y_pred, [1, -1])
      return tf.reduce_mean(metric(y_true, y_pred))
  return with_binary_averaging
