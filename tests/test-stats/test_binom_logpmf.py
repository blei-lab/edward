from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import binom
from scipy import stats


class test_binom_logpmf_class(tf.test.TestCase):

  def _test(self, x, n, p):
    val_true = stats.binom.logpmf(x, n, p)
    with self.test_session():
      self.assertAllClose(binom.logpmf(x, n=n, p=p).eval(), val_true)

  def test_0d(self):
    self._test(np.array(0, dtype=np.float32),
               np.array(1, dtype=np.float32),
               0.5)
    self._test(np.array(1, dtype=np.float32),
               np.array(2, dtype=np.float32),
               0.75)

  def test_1d(self):
    self._test(np.array([0, 1, 0], dtype=np.float32),
               np.array(1, dtype=np.float32),
               0.5)
    self._test(np.array([1, 0, 0], dtype=np.float32),
               np.array(1, dtype=np.float32),
               0.75)

  def test_2d(self):
    self._test(np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32),
               np.array(1, dtype=np.float32),
               0.5)
    self._test(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
               np.array(1, dtype=np.float32),
               0.75)

if __name__ == '__main__':
  tf.test.main()
