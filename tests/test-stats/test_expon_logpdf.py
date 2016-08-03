from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import expon
from scipy import stats

class test_expon_logpdf_class(tf.test.TestCase):


    def _test(self, x, scale=1):
        xtf = tf.constant(x)
        val_true = stats.expon.logpdf(x, scale=scale)
        with self.test_session():
            self.assertAllClose(expon.logpdf(xtf, scale=tf.constant(scale)).eval(), val_true)


    def test_0d(self):
        self._test(0.3)
        self._test(0.7)

        self._test(0.3, scale=1.0)
        self._test(0.7, scale=1.0)

        self._test(0.3, scale=0.5)
        self._test(0.7, scale=0.5)

        self._test(0.3, scale=5.0)
        self._test(0.7, scale=5.0)


    def test_1d(self):
        self._test([0.5, 2.3, 5.8, 10.1], scale=5.0)


    def test_2d(self):
        self._test(np.array([[0.5, 2.3, 5.8, 10.1],[0.5, 2.3, 5.8, 10.1]]), scale=5.0)
