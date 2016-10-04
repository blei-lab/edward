from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Exponential
from edward.util import get_dims


def _test(lam, n):
  x = Exponential(lam=lam)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(lam)
  assert val_est == val_true


class test_exponential_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      _test(2.0, [1])
      _test(np.array(2.0), [1])
      _test(tf.constant(2.0), [1])

  def test_1d(self):
    with self.test_session():
      _test(np.array([2.0]), [1])
      _test(np.array([2.0]), [5])
      _test(np.array([2.0, 8.0]), [1])
      _test(np.array([2.0, 8.0]), [10])
      _test(tf.constant([2.0]), [1])
      _test(tf.constant([2.0]), [5])
      _test(tf.constant([2.0, 8.0]), [1])
      _test(tf.constant([2.0, 8.0]), [10])

if __name__ == '__main__':
  tf.test.main()
