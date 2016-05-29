from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import nbinom
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpmf(x, n, p):
    xtf = tf.constant(x)
    val_true = stats.nbinom.logpmf(x, n, p)
    _assert_eq(nbinom.logpmf(xtf, n, p), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant(n), tf.constant(p)), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant(n), tf.constant([p])), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant([n]), tf.constant(p)), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant([n]), tf.constant([p])), val_true)

def test_logpmf_int_scalar():
    _test_logpmf(1, 5, 0.5)
    _test_logpmf(2, 5, 0.75)

def test_logpmf_float_scalar():
    _test_logpmf(1.0, 5, 0.5)
    _test_logpmf(2.0, 5, 0.75)

def test_logpmf_int_1d():
    _test_logpmf([1, 5, 3], 5, 0.5)
    _test_logpmf([2, 8, 2], 5, 0.75)

def test_logpmf_float_1d():
    _test_logpmf([1.0, 5.0, 3.0], 5, 0.5)
    _test_logpmf([2.0, 8.0, 2.0], 5, 0.75)

def test_logpmf_int_2d():
    _test_logpmf(np.array([[1, 5, 3],[2, 8, 2]]), 5, 0.5)
    _test_logpmf(np.array([[2, 8, 2],[1, 5, 3]]), 5, 0.75)

def test_logpmf_float_2d():
    _test_logpmf(np.array([[1.0, 5.0, 3.0],[2.0, 8.0, 2.0]]), 5, 0.5)
    _test_logpmf(np.array([[2.0, 8.0, 2.0],[1.0, 5.0, 3.0]]), 5, 0.75)
