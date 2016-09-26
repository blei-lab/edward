from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import InverseGamma
from edward.util import get_dims


def _test(alpha, beta, n):
  x = InverseGamma(alpha=alpha, beta=beta)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(alpha)
  assert val_est == val_true


class test_inversegamma_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      _test(0.5, 0.5, [1])
      _test(np.array(0.5), np.array(0.5), [1])
      _test(tf.constant(0.5), tf.constant(0.5), [1])

  def test_1d(self):
    with self.test_session():
      _test(np.array([0.5]), np.array([0.5]), [1])
      _test(np.array([0.5]), np.array([0.5]), [5])
      _test(np.array([0.2, 0.8]), np.array([0.2, 0.8]), [1])
      _test(np.array([0.2, 0.8]), np.array([0.2, 0.8]), [10])
      _test(tf.constant([0.5]), tf.constant([0.5]), [1])
      _test(tf.constant([0.5]), tf.constant([0.5]), [5])
      _test(tf.constant([0.2, 0.8]), tf.constant([0.2, 0.8]), [1])
      _test(tf.constant([0.2, 0.8]), tf.constant([0.2, 0.8]), [10])

if __name__ == '__main__':
  tf.test.main()
