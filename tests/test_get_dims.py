from __future__ import print_function
import tensorflow as tf

from edward.util import get_dims

def test_scalar():
    x = tf.constant(0.0)
    assert get_dims(x) == [1]

def test_1d():
    x = tf.zeros([2])
    assert get_dims(x) == [2]

def test_2d():
    x = tf.zeros([2, 2])
    assert get_dims(x) == [2, 2]
