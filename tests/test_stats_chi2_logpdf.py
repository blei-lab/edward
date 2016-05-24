from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import chi2
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpdf(x, df):
    xtf = tf.constant(x)
    val_true = stats.chi2.logpdf(x, df)
    _assert_eq(chi2.logpdf(xtf, df), val_true)
    _assert_eq(chi2.logpdf(xtf, tf.constant(df)), val_true)
    _assert_eq(chi2.logpdf(xtf, tf.constant([df])), val_true)

def test_logpdf_scalar():
    _test_logpdf(0.2, df=2)
    _test_logpdf(0.623, df=2)

def test_logpdf_1d():
    _test_logpdf([0.1, 1.0, 0.58, 2.3], df=3)

def test_logpdf_2d():
    _test_logpdf(np.array([[0.1, 1.0, 0.58, 2.3],[0.3, 1.1, 0.68, 1.2]]), df=3)
