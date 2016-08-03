from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import norm
from scipy import stats

class test_norm_logpdf_class(tf.test.TestCase):

    def _test(self, x, loc=0, scale=1):
        xtf = tf.constant(x)
        val_true = stats.norm.logpdf(x, loc, scale)
        with self.test_session():
            self.assertAllClose(norm.logpdf(xtf, loc, scale).eval(), val_true)
            self.assertAllClose(norm.logpdf(xtf, tf.constant(loc), tf.constant(scale)).eval(), val_true)
        

    def test_0d(self):
        self._test(0.0)
        self._test(0.623)


    def test_1d(self):
        self._test([0.0, 1.0, 0.58, 2.3])


    def test_2d(self):
        self._test(np.array([[0.0, 1.0, 0.58, 2.3], [0.1, 1.5, 4.18, 0.3]]))
