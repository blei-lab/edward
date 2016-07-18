from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import uniform
from scipy import stats

sess = tf.Session()


def uniform_entropy_vec(loc, scale):
    """Vectorized version of stats.uniform.entropy."""
    if isinstance(loc, float) or isinstance(loc, int):
        return stats.uniform.entropy(loc, scale)
    else:
        return np.array([stats.uniform.entropy(loc_x, scale_x)
                         for loc_x,scale_x in zip(loc,scale)])


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(loc=0, scale=1):
    val_true = uniform_entropy_vec(loc, scale)
    _assert_eq(uniform.entropy(loc, scale), val_true)
    _assert_eq(uniform.entropy(tf.constant(loc), tf.constant(scale)), val_true)
    _assert_eq(uniform.entropy(tf.constant([loc]), tf.constant(scale)), val_true)
    _assert_eq(uniform.entropy(tf.constant(loc), tf.constant([scale])), val_true)
    _assert_eq(uniform.entropy(tf.constant([loc]), tf.constant([scale])), val_true)


def test_0d():
    _test()
    _test(loc=1.0, scale=1.0)
    _test(loc=0.5, scale=5.0)
    _test(loc=5.0, scale=0.5)


def test_1d():
    _test([0.5, 0.3, 0.8, 0.2], [0.5, 0.3, 0.8, 0.2])
