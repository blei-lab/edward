from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import hessian

sess = tf.Session()


def _test(y, xs, val_true):
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)
        val_est = hessian(y, xs).eval()
        assert np.all(val_est == val_true)


def test_0d():
    x1 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
    x2 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
    y = tf.pow(x1, tf.constant(2.0)) + tf.constant(2.0) * x1 * x2 + \
        tf.constant(3.0) * tf.pow(x2, tf.constant(2.0)) + \
        tf.constant(4.0) * x1 + tf.constant(5.0) * x2 + tf.constant(6.0)
    _test(y, [x1], val_true=np.array([[2.0]]))
    _test(y, [x2], val_true=np.array([[6.0]]))


def test_1d():
    x1 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
    x2 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
    y = tf.pow(x1, tf.constant(2.0)) + tf.constant(2.0) * x1 * x2 + \
        tf.constant(3.0) * tf.pow(x2, tf.constant(2.0)) + \
        tf.constant(4.0) * x1 + tf.constant(5.0) * x2 + tf.constant(6.0)
    _test(y, [x1, x2], val_true=np.array([[2.0, 2.0], [2.0, 6.0]]))
    x3 = tf.Variable(tf.random_normal([3], dtype=tf.float32))
    y = tf.pow(x2, tf.constant(2.0)) + tf.reduce_sum(x3)
    _test(y, [x3], val_true=np.zeros([3, 3]))
    _test(y, [x2, x3], val_true=np.diag([2.0, 0.0, 0.0, 0.0]))


def test_2d():
    x1 = tf.Variable(tf.random_normal([3, 2], dtype=tf.float32))
    x2 = tf.Variable(tf.random_normal([2], dtype=tf.float32))
    y = tf.reduce_sum(tf.pow(x1, tf.constant(2.0))) + tf.reduce_sum(x2)
    _test(y, [x1], val_true=np.diag([2.0]*6))
    _test(y, [x1, x2], val_true=np.diag([2.0]*6+[0.0]*2))
