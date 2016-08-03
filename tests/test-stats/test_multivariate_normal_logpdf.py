from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal
from scipy import stats

class test_multivariate_normal_logpdf_class(tf.test.TestCase):

    def _test(self, x, mean=None, cov=1):
        xtf = tf.constant(x)
        mean_tf = tf.convert_to_tensor(mean)
        cov_tf = tf.convert_to_tensor(cov)
        val_true = stats.multivariate_normal.logpdf(x, mean, cov)
        with self.test_session():
            self.assertAllClose(multivariate_normal.logpdf(xtf, mean, cov).eval(), val_true)
            self.assertAllClose(multivariate_normal.logpdf(xtf, mean_tf, cov).eval(), val_true)
            self.assertAllClose(multivariate_normal.logpdf(xtf, mean, cov_tf).eval(), val_true)
            self.assertAllClose(multivariate_normal.logpdf(xtf, mean_tf, cov_tf).eval(), val_true)


    def test_int_1d(self):
        x = [0, 0]
        self._test(x, np.zeros([2]), np.ones([2]))
        self._test(x, np.zeros(2), np.diag(np.ones(2)))
        xtf = tf.constant(x)
        val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
        with self.test_session():
            self.assertAllClose(multivariate_normal.logpdf(xtf).eval(), val_true)

        self._test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))


    def test_float_1d(self):
        x = [0.0, 0.0]
        self._test(x, np.zeros([2]), np.ones([2]))
        self._test(x, np.zeros(2), np.diag(np.ones(2)))
        xtf = tf.constant(x)
        val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
        with self.test_session():
            self.assertAllClose(multivariate_normal.logpdf(xtf).eval(), val_true)

        self._test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))


    def test_float_2d(self):
        x = np.array([[0.3, 0.7],[0.2, 0.8]])
        self._test(x, np.zeros([2]), np.ones([2]))
        self._test(x, np.zeros(2), np.diag(np.ones(2)))
        xtf = tf.constant(x)
        val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
        with self.test_session():
            self.assertAllClose(multivariate_normal.logpdf(xtf).eval(), val_true)

        self._test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))
