from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import lognorm
from scipy import stats


class test_lognorm_logpdf_class(tf.test.TestCase):

  def _test(self, x, s):
    val_true = stats.lognorm.logpdf(x, s)
    with self.test_session():
      self.assertAllClose(lognorm.logpdf(x, s=s).eval(), val_true)

  def test_0d(self):
    self._test(2.0, s=1)
    self._test(0.623, s=1)

  def test_1d(self):
    self._test([2.0, 1.0, 0.58, 2.3], s=1)

  def test_2d(self):
    self._test(np.array([[2.0, 1.0, 0.58, 2.3], [2.1, 1.3, 1.58, 0.3]]),
               s=1)

if __name__ == '__main__':
  tf.test.main()
