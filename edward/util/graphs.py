from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import sys
import tensorflow as tf

from edward.models.random_variable import _RANDOM_VARIABLE_COLLECTION


def get_session():
  """Get the globally defined TensorFlow session.

  If the session is not already defined, then the function will create
  a global session.

  Returns:
    _ED_SESSION: tf.InteractiveSession.
  """
  global _ED_SESSION
  if tf.get_default_session() is None:
    _ED_SESSION = tf.InteractiveSession()
  else:
    _ED_SESSION = tf.get_default_session()

  save_stderr = sys.stderr
  try:
    import os
    sys.stderr = open(os.devnull, 'w')  # suppress keras import
    from keras import backend as K
    sys.stderr = save_stderr
    have_keras = True
  except ImportError:
    sys.stderr = save_stderr
    have_keras = False
  if have_keras:
    K.set_session(_ED_SESSION)

  return _ED_SESSION


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


def set_seed(x):
  """Set seed for both NumPy and TensorFlow.

  Args:
    x: int, float.
      seed
  """
  node_names = list(six.iterkeys(tf.get_default_graph()._nodes_by_name))
  if len(node_names) > 0 and node_names != ['keras_learning_phase']:
    raise RuntimeError("Seeding is not supported after initializing "
                       "part of the graph. "
                       "Please move set_seed to the beginning of your code.")

  np.random.seed(x)
  tf.set_random_seed(x)
