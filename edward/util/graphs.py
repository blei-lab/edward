from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import _RANDOM_VARIABLE_COLLECTION


def random_variables(graph=None):
  """Return all random variables in the TensorFlow graph.

  Args:
    graph: TensorFlow graph.

  Returns:
    list of RandomVariable.
  """
  if graph is None:
    graph = tf.get_default_graph()

  return _RANDOM_VARIABLE_COLLECTION[graph]
