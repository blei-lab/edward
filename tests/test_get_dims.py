from __future__ import print_function
import tensorflow as tf

from blackbox.util import get_dims


def test_get_dims_scalar():
    x = tf.constant(0.0)
    assert get_dims(x) == [1]


def test_get_dims_1d():
    x = tf.zeros([2])
    assert get_dims(x) == [2]


def test_get_dims_2d():
    x = tf.zeros([2, 2])
    assert get_dims(x) == [2, 2]
