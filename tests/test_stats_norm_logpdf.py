from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import norm
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        print(val_ed.eval())
        print(val_true)
        assert np.allclose(val_ed.eval(), val_true)

def _test_logpdf(x, loc=0, scale=1):
    xtf = tf.constant(x)
    val_true = stats.norm.logpdf(x, loc, scale)
    _assert_eq(norm.logpdf(xtf, loc, scale), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant(loc), tf.constant(scale)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant([loc]), tf.constant(scale)), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant(loc), tf.constant([scale])), val_true)
    _assert_eq(norm.logpdf(xtf, tf.constant([loc]), tf.constant([scale])), val_true)

def test_logpdf_scalar():
    _test_logpdf(0.0)
    _test_logpdf(0.623)

def test_logpdf_1d():
    _test_logpdf([0.0, 1.0, 0.58, 2.3])
