from __future__ import print_function
import tensorflow as tf

from blackbox.util import trace

sess = tf.Session()


def test_trace_scalar():
    X = tf.diag([2])
    with sess.as_default():
        assert trace(X).eval() == 2


def test_trace_mat():
    X = tf.diag(tf.ones([2]))
    with sess.as_default():
        assert trace(X).eval() == 2
