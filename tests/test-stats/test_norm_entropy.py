from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import norm
from scipy import stats

sess = tf.Session()


def norm_entropy_vec(loc, scale):
    """Vectorized version of stats.norm.entropy."""
    if isinstance(loc, float) or isinstance(loc, int):
        return stats.norm.entropy(loc, scale)
    else:
        return np.array([stats.norm.entropy(loc_x, scale_x)
                         for loc_x,scale_x in zip(loc,scale)])


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(loc, scale):
    val_true = norm_entropy_vec(loc, scale)
    _assert_eq(norm.entropy(loc, scale), val_true)
    _assert_eq(norm.entropy(tf.constant(loc), tf.constant(scale)), val_true)
    _assert_eq(norm.entropy(tf.constant([loc]), tf.constant(scale)), val_true)
    _assert_eq(norm.entropy(tf.constant(loc), tf.constant([scale])), val_true)
    _assert_eq(norm.entropy(tf.constant([loc]), tf.constant([scale])), val_true)


def test_empty():
    _assert_eq(norm.entropy(), stats.norm.entropy())


def test_0d():
    _test(1.0, 1.0)
    _test(1.0, 1.0)

    _test(0.5, 5.0)
    _test(5.0, 0.5)


def test_1d():
    _test([0.5, 1.2, 5.3, 8.7], [0.5, 1.2, 5.3, 8.7])
