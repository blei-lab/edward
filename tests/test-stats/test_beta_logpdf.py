from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import beta
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, a, b):
    xtf = tf.constant(x)
    val_true = stats.beta.logpdf(x, a, b)
    _assert_eq(beta.logpdf(xtf, a, b), val_true)
    _assert_eq(beta.logpdf(xtf, tf.constant(a), tf.constant(b)), val_true)
    _assert_eq(beta.logpdf(xtf, tf.constant([a]), tf.constant(b)), val_true)
    _assert_eq(beta.logpdf(xtf, tf.constant(a), tf.constant([b])), val_true)
    _assert_eq(beta.logpdf(xtf, tf.constant([a]), tf.constant([b])), val_true)


def test_0d():
    _test(0.3, a=0.5, b=0.5)
    _test(0.7, a=0.5, b=0.5)

    _test(0.3, a=1.0, b=1.0)
    _test(0.7, a=1.0, b=1.0)

    _test(0.3, a=0.5, b=5.0)
    _test(0.7, a=0.5, b=5.0)

    _test(0.3, a=5.0, b=0.5)
    _test(0.7, a=5.0, b=0.5)


def test_1d():
    _test([0.5, 0.3, 0.8, 0.1], a=0.5, b=0.5)


def test_2d():
    _test(np.array([[0.5, 0.3, 0.8, 0.1],[0.1, 0.7, 0.2, 0.4]]),
                 a=0.5, b=0.5)
