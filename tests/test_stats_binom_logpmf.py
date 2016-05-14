from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import binom
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpmf(x, n, p):
    xtf = tf.constant(x)
    val_true = stats.binom.logpmf(x, n, p)
    _assert_eq(binom.logpmf(xtf, n, p), val_true)
    _assert_eq(binom.logpmf(xtf, tf.constant(n), tf.constant(p)), val_true)
    _assert_eq(binom.logpmf(xtf, tf.constant([n]), tf.constant([p])), val_true)

def test_logpmf_int_scalar():
    _test_logpmf(0, 1, 0.5)
    _test_logpmf(1, 2, 0.75)

def test_logpmf_float_scalar():
    _test_logpmf(0.0, 1, 0.5)
    _test_logpmf(1.0, 2, 0.75)

def test_logpmf_int_1d():
    _test_logpmf([0, 1, 0], 1, 0.5)
    _test_logpmf([1, 0, 0], 1, 0.75)

def test_logpmf_float_1d():
    _test_logpmf([0.0, 1.0, 0.0], 1, 0.5)
    _test_logpmf([1.0, 0.0, 0.0], 1, 0.75)
