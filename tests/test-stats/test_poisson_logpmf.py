from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import poisson
from scipy import stats


class test_poisson_logpmf_class(tf.test.TestCase):

  def _test(self, x, lam):
    val_true = stats.poisson.logpmf(x, lam)
    with self.test_session():
      self.assertAllClose(poisson.logpmf(x, lam=lam).eval(), val_true)

  def test_0d(self):
    self._test(np.array(0.0, dtype=np.float32), 0.5)
    self._test(np.array(1.0, dtype=np.float32), 0.75)

  def test_1d(self):
    self._test(np.array([0.0, 1.0, 3.0], dtype=np.float32), 0.5)
    self._test(np.array([1.0, 8.0, 2.0], dtype=np.float32), 0.75)

  def test_2d(self):
    self._test(np.array([[0.0, 1.0, 3.0], [0.0, 1.0, 3.0]], dtype=np.float32),
               0.5)
    self._test(np.array([[1.0, 8.0, 2.0], [1.0, 8.0, 2.0]], dtype=np.float32),
               0.75)

if __name__ == '__main__':
  tf.test.main()
