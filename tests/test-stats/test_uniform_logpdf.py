from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import uniform
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, loc=0, scale=1):
    xtf = tf.constant(x)
    val_true = stats.uniform.logpdf(x, loc, scale)
    _assert_eq(uniform.logpdf(xtf, loc, scale), val_true)
    _assert_eq(uniform.logpdf(xtf, tf.constant(loc), tf.constant(scale)), val_true)
    _assert_eq(uniform.logpdf(xtf, tf.constant([loc]), tf.constant(scale)), val_true)
    _assert_eq(uniform.logpdf(xtf, tf.constant(loc), tf.constant([scale])), val_true)
    _assert_eq(uniform.logpdf(xtf, tf.constant([loc]), tf.constant([scale])), val_true)


def test_0d():
    _test(0.3)
    _test(0.7)

    _test(1.3, loc=1.0, scale=1.0)
    _test(1.7, loc=1.0, scale=1.0)

    _test(2.3, loc=0.5, scale=5.0)
    _test(2.7, loc=0.5, scale=5.0)

    _test(5.3, loc=5.0, scale=0.5)
    _test(5.1, loc=5.0, scale=0.5)


def test_1d():
    _test([0.5, 0.3, 0.8, 0.2], loc=0.1, scale=0.9)


def test_2d():
    _test(np.array([[0.5, 0.3, 0.8, 0.2],[0.5, 0.3, 0.8, 0.2]]),
                 loc=0.1, scale=0.9)
