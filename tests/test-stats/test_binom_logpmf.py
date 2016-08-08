from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import binom
from scipy import stats

class test_binom_logpmf_class(tf.test.TestCase):

    def _test(self, x, n, p):
        xtf = tf.constant(x)
        val_true = stats.binom.logpmf(x, n, p)
        with self.test_session():
            self.assertAllClose(binom.logpmf(xtf, n, p).eval(), val_true)
            self.assertAllClose(binom.logpmf(xtf, tf.constant(n), tf.constant(p)).eval(), val_true)


    def test_int_0d(self):
        self._test(0, 1, 0.5)
        self._test(1, 2, 0.75)


    def test_float_0d(self):
        self._test(0.0, 1, 0.5)
        self._test(1.0, 2, 0.75)


    def test_int_1d(self):
        self._test([0, 1, 0], 1, 0.5)
        self._test([1, 0, 0], 1, 0.75)


    def test_float_1d(self):
        self._test([0.0, 1.0, 0.0], 1, 0.5)
        self._test([1.0, 0.0, 0.0], 1, 0.75)


    def test_int_2d(self):
        self._test(np.array([[0, 1, 0],[1, 0, 0]]), 1, 0.5)
        self._test(np.array([[1, 0, 0],[0, 1, 0]]), 1, 0.75)


    def test_float_2d(self):
        self._test(np.array([[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]]), 1, 0.5)
        self._test(np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]]), 1, 0.75)
