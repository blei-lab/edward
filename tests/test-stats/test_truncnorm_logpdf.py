from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import truncnorm
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, a, b, loc=0, scale=1):
    xtf = tf.constant(x)
    val_true = stats.truncnorm.logpdf(x, a, b, loc, scale)
    _assert_eq(truncnorm.logpdf(xtf, a, b, loc, scale), val_true)
    _assert_eq(truncnorm.logpdf(xtf, a, b, tf.constant(loc), tf.constant(scale)), val_true)
    _assert_eq(truncnorm.logpdf(xtf, a, b, tf.constant([loc]), tf.constant(scale)), val_true)
    _assert_eq(truncnorm.logpdf(xtf, a, b, tf.constant(loc), tf.constant([scale])), val_true)
    _assert_eq(truncnorm.logpdf(xtf, a, b, tf.constant([loc]), tf.constant([scale])), val_true)


def test_0d():
    _test(0.0, a=-1.0, b=3.0)
    _test(0.623, a=-1.0, b=3.0)


def test_1d():
    _test([0.0, 1.0, 0.58, 2.3], a=-1.0, b=3.0)


def test_2d():
    _test(np.array([[0.0, 1.0, 0.58, 2.3],[0.0, 1.0, 0.58, 2.3]]),
                 a=-1.0, b=3.0)
