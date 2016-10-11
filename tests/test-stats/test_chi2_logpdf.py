from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import chi2
from scipy import stats


class test_chi2_logpdf_class(tf.test.TestCase):

  def _test(self, x, df):
    val_true = stats.chi2.logpdf(x, df)
    with self.test_session():
      self.assertAllClose(chi2.logpdf(x, df=df).eval(), val_true)

  def test_0d(self):
    self._test(0.2, df=2)
    self._test(0.623, df=2)

  def test_1d(self):
    self._test([0.1, 1.0, 0.58, 2.3], df=3)

  def test_2d(self):
    self._test(np.array([[0.1, 1.0, 0.58, 2.3], [0.3, 1.1, 0.68, 1.2]]), df=3)

if __name__ == '__main__':
  tf.test.main()
