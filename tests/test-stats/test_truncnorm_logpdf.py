from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import truncnorm
from scipy import stats


class test_trucnorm_logpdf_class(tf.test.TestCase):

  def _test(self, x, a, b, loc=0, scale=1):
    xtf = tf.constant(x)
    val_true = stats.truncnorm.logpdf(x, a, b, loc, scale)
    with self.test_session():
      self.assertAllClose(truncnorm.logpdf(xtf, a, b, loc, scale).eval(),
                          val_true)
      self.assertAllClose(truncnorm.logpdf(xtf, a, b, tf.constant(loc),
                                           tf.constant(scale)).eval(), val_true)

  def test_0d(self):
    self._test(0.0, a=-1.0, b=3.0)
    self._test(0.623, a=-1.0, b=3.0)

  def test_1d(self):
    self._test([0.0, 1.0, 0.58, 2.3], a=-1.0, b=3.0)

  def test_2d(self):
    self._test(np.array([[0.0, 1.0, 0.58, 2.3], [0.0, 1.0, 0.58, 2.3]]),
               a=-1.0, b=3.0)
