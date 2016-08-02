from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import t
from scipy import stats

class test_t_logpdf_class(tf.test.TestCase):

    def _test(self, x, df, loc=0, scale=1):
        xtf = tf.constant(x)
        val_true = stats.t.logpdf(x, df, loc, scale)
        with self.test_session():
            self.assertAllClose(t.logpdf(xtf, df, loc, scale).eval(), val_true)
            self.assertAllClose(t.logpdf(xtf, df, tf.constant(loc), tf.constant(scale)).eval(), val_true)
    
    
    def test_0d(self):
        self._test(0.0, df=3)
        self._test(0.623, df=3)


    def test_1d(self):
        self._test([0.0, 1.0, 0.58, 2.3], df=3)


    def test_2d(self):
        self._test(np.array([[0.0, 1.0, 0.58, 2.3],[0.0, 1.0, 0.58, 2.3]]), df=3)
