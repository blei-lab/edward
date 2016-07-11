from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import get_dims


def _test(x, val_true):
    val_est = get_dims(x)
    assert val_est == val_true


def test_0d():
    x = tf.constant(0.0)
    _test(x, [])
    x = np.array(0.0)
    _test(x, [])


def test_1d():
    x = tf.zeros([2])
    _test(x, [2])
    x = np.zeros([2])
    _test(x, [2])


def test_2d():
    x = tf.zeros([2, 2])
    _test(x, [2, 2])
    x = np.zeros([2, 2])
    _test(x, [2, 2])
