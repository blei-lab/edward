from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import gamma
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpdf(x, a=0.5, b=0.5):
    xtf = tf.constant(x)
    val_true = stats.gamma.logpdf(x, a, scale=b)
    _assert_eq(gamma.logpdf(xtf, tf.constant(a), tf.constant(b)), val_true)
    _assert_eq(gamma.logpdf(xtf, tf.constant([a]), tf.constant(b)), val_true)
    _assert_eq(gamma.logpdf(xtf, tf.constant(a), tf.constant([b])), val_true)
    _assert_eq(gamma.logpdf(xtf, tf.constant([a]), tf.constant([b])), val_true)

def test_logpdf_scalar():
    _test_logpdf(0.3)
    _test_logpdf(0.7)

    _test_logpdf(0.3, a=1.0, b=1.0)
    _test_logpdf(0.7, a=1.0, b=1.0)

    _test_logpdf(0.3, a=0.5, b=5.0)
    _test_logpdf(0.7, a=0.5, b=5.0)

    _test_logpdf(0.3, a=5.0, b=0.5)
    _test_logpdf(0.7, a=5.0, b=0.5)

def test_logpdf_1d():
    _test_logpdf([0.5, 1.2, 5.3, 8.7], a=0.5, b=0.5)
