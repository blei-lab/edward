from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)

def _test_logpdf(scalar, param):
    x = tf.constant(scalar)
    val_true = stats.bernoulli.logpmf(scalar, param)
    _assert_eq(bernoulli.logpmf(x, tf.constant(param)), val_true)
    _assert_eq(bernoulli.logpmf(x, tf.constant([param])), val_true)

def test_logpdf_int_scalar():
    _test_logpdf(0, 0.5)
    _test_logpdf(1, 0.75)

def test_logpdf_float_scalar():
    _test_logpdf(0.0, 0.5)
    _test_logpdf(1.0, 0.75)

def test_logpdf_int_1d():
    _test_logpdf([0], 0.5)
    _test_logpdf([1], 0.75)

def test_logpdf_float_1d():
    _test_logpdf([0.0], 0.5)
    _test_logpdf([1.0], 0.75)
