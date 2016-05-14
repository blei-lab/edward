from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import t
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpdf(x, df, loc=0, scale=1):
    xtf = tf.constant(x)
    val_true = stats.t.logpdf(x, df, loc, scale)
    _assert_eq(t.logpdf(xtf, df, loc, scale), val_true)
    _assert_eq(t.logpdf(xtf, df, tf.constant(loc), tf.constant(scale)), val_true)
    _assert_eq(t.logpdf(xtf, df, tf.constant([loc]), tf.constant(scale)), val_true)
    _assert_eq(t.logpdf(xtf, df, tf.constant(loc), tf.constant([scale])), val_true)
    _assert_eq(t.logpdf(xtf, df, tf.constant([loc]), tf.constant([scale])), val_true)

def test_logpdf_scalar():
    _test_logpdf(0.0, df=3)
    _test_logpdf(0.623, df=3)

def test_logpdf_1d():
    _test_logpdf([0.0, 1.0, 0.58, 2.3], df=3)
