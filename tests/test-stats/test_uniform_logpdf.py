from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import uniform
from scipy import stats


class test_uniform_logpdf_class(tf.test.TestCase):

  def _test(self, x, a, b):
    val_true = stats.uniform.logpdf(x, a, b - a)
    with self.test_session():
      self.assertAllClose(uniform.logpdf(x, a=a, b=b).eval(), val_true)

  def test_0d(self):
    self._test(0.3, a=0.0, b=1.0)
    self._test(0.7, a=0.0, b=1.0)

    self._test(1.3, a=1.0, b=2.0)
    self._test(1.7, a=1.0, b=2.0)

    self._test(2.3, a=0.5, b=5.0)
    self._test(2.7, a=0.5, b=5.0)

    self._test(5.3, a=5.0, b=5.5)
    self._test(5.1, a=5.0, b=5.5)

  def test_1d(self):
    self._test(np.array([0.5, 0.3, 0.8, 0.2], dtype=np.float32), a=0.1, b=0.9)

  def test_2d(self):
    self._test(np.array([[0.5, 0.3, 0.8, 0.2], [0.5, 0.3, 0.8, 0.2]],
                        dtype=np.float32),
               a=0.1, b=0.9)

if __name__ == '__main__':
  tf.test.main()
