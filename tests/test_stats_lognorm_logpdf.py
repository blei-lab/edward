from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import lognorm
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)

def _test_logpdf(x, s=1):
    xtf = tf.constant(x)
    val_true = stats.lognorm.logpdf(x, s)
    _assert_eq(lognorm.logpdf(xtf, s), val_true)
    _assert_eq(lognorm.logpdf(xtf, tf.constant(s)), val_true)
    _assert_eq(lognorm.logpdf(xtf, tf.constant([s])), val_true)

def test_logpdf_scalar():
    _test_logpdf(2.0)
    _test_logpdf(0.623)

def test_logpdf_1d():
    _test_logpdf([2.0, 1.0, 0.58, 2.3])
