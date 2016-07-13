from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import poisson
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, mu):
    xtf = tf.constant(x)
    val_true = stats.poisson.logpmf(x, mu)
    _assert_eq(poisson.logpmf(xtf, mu), val_true)
    _assert_eq(poisson.logpmf(xtf, tf.constant(mu)), val_true)
    _assert_eq(poisson.logpmf(xtf, tf.constant([mu])), val_true)


def test_int_0d():
    _test(0, 0.5)
    _test(1, 0.75)


def test_float_0d():
    _test(0.0, 0.5)
    _test(1.0, 0.75)


def test_int_1d():
    _test([0, 1, 3], 0.5)
    _test([1, 8, 2], 0.75)


def test_float_1d():
    _test([0.0, 1.0, 3.0], 0.5)
    _test([1.0, 8.0, 2.0], 0.75)


def test_int_2d():
    _test(np.array([[0, 1, 3],[1, 8, 2]]), 0.5)
    _test(np.array([[1, 8, 2],[0, 1, 3]]), 0.75)


def test_float_2d():
    _test(np.array([[0.0, 1.0, 3.0],[0.0, 1.0, 3.0]]), 0.5)
    _test(np.array([[1.0, 8.0, 2.0],[1.0, 8.0, 2.0]]), 0.75)
