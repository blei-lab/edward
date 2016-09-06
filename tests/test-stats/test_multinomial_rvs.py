from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multinomial
from scipy import stats


class test_multinomial_rvs_class(tf.test.TestCase):

  def _test(self, n, p, size):
    val_est = multinomial.rvs(n, p, size=size).shape
    val_true = (size, ) + np.asarray(p).shape
    assert val_est == val_true

  def test_1d(self):
    self._test(3, np.array([0.4, 0.6]), 1)
    self._test(np.array(3), np.array([0.4, 0.6]), 5)

  # def test_2d(self):
  #    self._test(np.array([3]), np.array([[0.4, 0.6]]), 1)
  #    self._test(np.array([3]), np.array([[0.4, 0.6]]), 5)
  #    self._test(np.array([3, 2]), np.array([[0.2, 0.8], [0.6, 0.4]]), 1)
  #    self._test(np.array([3, 2]), np.array([[0.2, 0.8], [0.6, 0.4]]), 10)

if __name__ == '__main__':
  tf.test.main()
