from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import t
from scipy import stats


class test_t_logpdf_class(tf.test.TestCase):

  def _test(self, x, df, mu, sigma):
    xtf = tf.constant(x)
    val_true = stats.t.logpdf(x, df, mu, sigma)
    with self.test_session():
      self.assertAllClose(t.logpdf(xtf, df, mu, sigma).eval(), val_true)
      self.assertAllClose(t.logpdf(xtf, df, tf.constant(mu),
                          tf.constant(sigma)).eval(), val_true)

  def test_0d(self):
    self._test(0.0, 3.0, 0.0, 1.0)
    self._test(0.623, 3.0, 0.0, 1.0)

  def test_1d(self):
    self._test([0.0, 1.0, 0.58, 2.3], 3.0, 0.0, 1.0)

  def test_2d(self):
    self._test(np.array([[0.0, 1.0, 0.58, 2.3], [0.0, 1.0, 0.58, 2.3]],
                        dtype=np.float32),
               3.0, np.array([0.0] * 4, dtype=np.float32),
               np.array([1.0] * 4, dtype=np.float32))
