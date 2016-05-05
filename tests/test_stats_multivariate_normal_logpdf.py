from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)

def _test_logpdf_standard_2d(x):
    xtf = tf.constant(x)
    val_true = stats.multivariate_normal.logpdf(x, np.zeros(2), np.diag(np.ones(2)))
    _assert_eq(multivariate_normal.logpdf(xtf), val_true)
    _assert_eq(multivariate_normal.logpdf(xtf, np.zeros([2]), np.ones([2])),
               val_true)
    _assert_eq(multivariate_normal.logpdf(xtf, tf.zeros([2]), tf.ones([2])),
               val_true)
    _assert_eq(
        multivariate_normal.logpdf(xtf, np.zeros([2]), np.diag(np.ones([2]))),
        val_true)

def test_logpdf_standard_float_2d():
    _test_logpdf_standard_2d([0.0, 0.0])

def test_logpdf_standard_int_2d():
    _test_logpdf_standard_2d([0, 0])

def test_logpdf_cov_float_2d():
    x = np.zeros(2)
    xtf = tf.constant([0.0, 0.0])
    val_true = stats.multivariate_normal.logpdf(
        x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))
    _assert_eq(multivariate_normal.logpdf(
        xtf, tf.zeros([2]), tf.constant([[2.0, 0.5], [0.5, 1.0]])),
        val_true)
