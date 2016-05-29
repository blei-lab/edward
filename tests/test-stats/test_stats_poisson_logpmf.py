from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import poisson
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpmf(x, mu):
    xtf = tf.constant(x)
    val_true = stats.poisson.logpmf(x, mu)
    _assert_eq(poisson.logpmf(xtf, mu), val_true)
    _assert_eq(poisson.logpmf(xtf, tf.constant(mu)), val_true)
    _assert_eq(poisson.logpmf(xtf, tf.constant([mu])), val_true)

def test_logpmf_int_scalar():
    _test_logpmf(0, 0.5)
    _test_logpmf(1, 0.75)

def test_logpmf_float_scalar():
    _test_logpmf(0.0, 0.5)
    _test_logpmf(1.0, 0.75)

def test_logpmf_int_1d():
    _test_logpmf([0, 1, 3], 0.5)
    _test_logpmf([1, 8, 2], 0.75)

def test_logpmf_float_1d():
    _test_logpmf([0.0, 1.0, 3.0], 0.5)
    _test_logpmf([1.0, 8.0, 2.0], 0.75)

def test_logpmf_int_2d():
    _test_logpmf(np.array([[0, 1, 3],[1, 8, 2]]), 0.5)
    _test_logpmf(np.array([[1, 8, 2],[0, 1, 3]]), 0.75)

def test_logpmf_float_2d():
    _test_logpmf(np.array([[0.0, 1.0, 3.0],[0.0, 1.0, 3.0]]), 0.5)
    _test_logpmf(np.array([[1.0, 8.0, 2.0],[1.0, 8.0, 2.0]]), 0.75)
