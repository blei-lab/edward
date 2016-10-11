from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import expon
from scipy import stats


class test_expon_logpdf_class(tf.test.TestCase):

  def _test(self, x, lam):
    val_true = stats.expon.logpdf(x, scale=1.0 / lam)
    with self.test_session():
      self.assertAllClose(expon.logpdf(x, lam=lam).eval(), val_true)

  def test_0d(self):
    self._test(0.3, lam=1.0)
    self._test(0.7, lam=1.0)

    self._test(0.3, lam=0.5)
    self._test(0.7, lam=0.5)

    self._test(0.3, lam=5.0)
    self._test(0.7, lam=5.0)

  def test_1d(self):
    self._test([0.5, 2.3, 5.8, 10.1], lam=5.0)

  def test_2d(self):
    self._test(np.array([[0.5, 2.3, 5.8, 10.1], [0.5, 2.3, 5.8, 10.1]],
                        dtype=np.float32),
               lam=5.0)

if __name__ == '__main__':
  tf.test.main()
