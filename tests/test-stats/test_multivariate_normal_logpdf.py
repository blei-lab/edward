from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, mean=None, cov=1):
    xtf = tf.constant(x)
    mean_tf = tf.convert_to_tensor(mean)
    cov_tf = tf.convert_to_tensor(cov)
    val_true = stats.multivariate_normal.logpdf(x, mean, cov)
    _assert_eq(multivariate_normal.logpdf(xtf, mean, cov), val_true)
    _assert_eq(multivariate_normal.logpdf(xtf, mean_tf, cov), val_true)
    _assert_eq(multivariate_normal.logpdf(xtf, mean, cov_tf), val_true)
    _assert_eq(multivariate_normal.logpdf(xtf, mean_tf, cov_tf), val_true)


def test_int_1d():
    x = [0, 0]
    _test(x, np.zeros([2]), np.ones([2]))
    _test(x, np.zeros(2), np.diag(np.ones(2)))
    xtf = tf.constant(x)
    val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
    _assert_eq(multivariate_normal.logpdf(xtf), val_true)

    _test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))


def test_float_1d():
    x = [0.0, 0.0]
    _test(x, np.zeros([2]), np.ones([2]))
    _test(x, np.zeros(2), np.diag(np.ones(2)))
    xtf = tf.constant(x)
    val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
    _assert_eq(multivariate_normal.logpdf(xtf), val_true)

    _test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))


def test_float_2d():
    x = np.array([[0.3, 0.7],[0.2, 0.8]])
    _test(x, np.zeros([2]), np.ones([2]))
    _test(x, np.zeros(2), np.diag(np.ones(2)))
    xtf = tf.constant(x)
    val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
    _assert_eq(multivariate_normal.logpdf(xtf), val_true)

    _test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))
