from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical
from scipy.special import gammaln


def categorical_logpmf(x, logits):
  """
  Parameters
  ----------
  x : np.ndarray
    int in domain {0, ..., K-1}
  logits : np.ndarray
    Vector of real-valued logit probabilities.
  """
  x = x.astype(np.int32)
  p = np.exp(logits) / np.sum(np.exp(logits), 0)
  return np.log(p)[x]


def categorical_logpmf_vec(x, logits):
  shape = x.shape
  if len(shape) == 0:
    return categorical_logpmf(x, logits)
  elif len(shape) == 1:
    size = shape[0]
    if len(logits.shape) == 1:
      return np.array([categorical_logpmf_vec(x[i], logits)
                       for i in range(size)])
    else:
      return np.array([categorical_logpmf_vec(x[i], logits[i, :])
                       for i in range(size)])
  elif len(shape) == 2:
    size = shape[0]
    return np.array([categorical_logpmf_vec(x[i, :], logits)
                     for i in range(size)])
  else:
    raise NotImplementedError()


def _test(logits, n):
  rv = Categorical(logits=logits)
  rv_sample = rv.sample(n)
  x = rv_sample.eval()
  x_tf = tf.constant(x, dtype=tf.int32)
  logits = logits.eval()
  assert np.allclose(rv.log_prob(x_tf).eval(),
                     categorical_logpmf_vec(x, logits))


class test_categorical_log_prob_class(tf.test.TestCase):

  def test_1d(self):
    ed.set_seed(98765)
    with self.test_session():
      _test(tf.constant([0.6, 0.4]), [1])
      _test(tf.constant([0.6, 0.4]), [2])

  def test_2d(self):
    ed.set_seed(98765)
    with self.test_session():
      _test(tf.constant([[0.5, 0.5], [0.6, 0.4]]), [1])
      _test(tf.constant([[0.5, 0.5], [0.6, 0.4]]), [2])
      _test(tf.constant([[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]]), [1])
      _test(tf.constant([[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]]), [2])

if __name__ == '__main__':
  tf.test.main()
