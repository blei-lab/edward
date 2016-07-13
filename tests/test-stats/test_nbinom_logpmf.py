from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import nbinom
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, n, p):
    xtf = tf.constant(x)
    val_true = stats.nbinom.logpmf(x, n, p)
    _assert_eq(nbinom.logpmf(xtf, n, p), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant(n), tf.constant(p)), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant(n), tf.constant([p])), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant([n]), tf.constant(p)), val_true)
    _assert_eq(nbinom.logpmf(xtf, tf.constant([n]), tf.constant([p])), val_true)


def test_int_0d():
    _test(1, 5, 0.5)
    _test(2, 5, 0.75)


def test_float_0d():
    _test(1.0, 5, 0.5)
    _test(2.0, 5, 0.75)


def test_int_1d():
    _test([1, 5, 3], 5, 0.5)
    _test([2, 8, 2], 5, 0.75)


def test_float_1d():
    _test([1.0, 5.0, 3.0], 5, 0.5)
    _test([2.0, 8.0, 2.0], 5, 0.75)


def test_int_2d():
    _test(np.array([[1, 5, 3],[2, 8, 2]]), 5, 0.5)
    _test(np.array([[2, 8, 2],[1, 5, 3]]), 5, 0.75)


def test_float_2d():
    _test(np.array([[1.0, 5.0, 3.0],[2.0, 8.0, 2.0]]), 5, 0.5)
    _test(np.array([[2.0, 8.0, 2.0],[1.0, 5.0, 3.0]]), 5, 0.75)
