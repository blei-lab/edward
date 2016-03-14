from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import norm

sess = tf.InteractiveSession()


def _assert_eq(res_bb, res_true):
    with sess.as_default():
        assert np.allclose(res_bb.eval(), res_true)


def test_entropy_scalar():
    x = tf.constant(1.0)
    _assert_eq(norm.entropy(x), 1.41894)


def test_entropy_1d():
    x = tf.ones([1])
    _assert_eq(norm.entropy(x), 1.41894)


def test_entropy_2d():
    x = tf.ones([2])
    _assert_eq(norm.entropy(x), 2.83788)
