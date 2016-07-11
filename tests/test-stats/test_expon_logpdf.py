from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import expon
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, scale=1):
    xtf = tf.constant(x)
    val_true = stats.expon.logpdf(x, scale=scale)
    _assert_eq(expon.logpdf(xtf, scale=tf.constant(scale)), val_true)
    _assert_eq(expon.logpdf(xtf, scale=tf.constant([scale])), val_true)


def test_0d():
    _test(0.3)
    _test(0.7)

    _test(0.3, scale=1.0)
    _test(0.7, scale=1.0)

    _test(0.3, scale=0.5)
    _test(0.7, scale=0.5)

    _test(0.3, scale=5.0)
    _test(0.7, scale=5.0)


def test_1d():
    _test([0.5, 2.3, 5.8, 10.1], scale=5.0)


def test_2d():
    _test(np.array([[0.5, 2.3, 5.8, 10.1],[0.5, 2.3, 5.8, 10.1]]), scale=5.0)
