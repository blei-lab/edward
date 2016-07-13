from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import invgamma
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(a, scale=1):
    val_true = stats.invgamma.entropy(a, scale=scale)
    _assert_eq(invgamma.entropy(a, scale), val_true)
    _assert_eq(invgamma.entropy(tf.constant(a), tf.constant(scale)), val_true)
    _assert_eq(invgamma.entropy(tf.constant([a]), tf.constant(scale)), val_true)
    _assert_eq(invgamma.entropy(tf.constant(a), tf.constant([scale])), val_true)
    _assert_eq(invgamma.entropy(tf.constant([a]), tf.constant([scale])), val_true)


def test_0d():
    _test(a=1.0, scale=1.0)
    _test(a=0.5, scale=5.0)
    _test(a=5.0, scale=0.5)


def test_1d():
    _test([0.5, 1.2, 5.3, 8.7], [0.5, 1.2, 5.3, 8.7])
