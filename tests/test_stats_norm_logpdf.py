from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import norm
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)

def _test_logpdf_scalar(x):
    xtf = tf.constant(x)
    val_true = stats.norm.logpdf(x)
    _assert_eq(norm.logpdf(xtf), val_true)
    _assert_eq(norm.logpdf(xtf, tf.zeros([1]), tf.constant(1.0)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.zeros([1]), tf.ones([1])), val_true)
    _assert_eq(norm.logpdf(xtf, tf.zeros([1]), tf.diag(tf.ones([1]))), val_true)

def test_logpdf_scalar():
    _test_logpdf_scalar(0.0)
    _test_logpdf_scalar(0.623)

def test_logpdf_1d():
    x = [0.0]
    xtf = tf.constant([0.0])
    val_true = stats.norm.logpdf(x)
    _assert_eq(norm.logpdf(xtf), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant(0.0), tf.constant(1.0)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant([0.0]), tf.constant(1.0)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant([0.0]), tf.constant([1.0])), val_true)

def test_logpdf_1by1mat():
    x = [[0.0]]
    xtf = tf.constant([[0.0]])
    val_true = stats.norm.logpdf(x)
    _assert_eq(norm.logpdf(xtf), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant(0.0), tf.constant(1.0)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant([0.0]), tf.constant(1.0)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant(0.0), tf.constant([1.0])), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant([0.0]), tf.constant([1.0])), val_true)
