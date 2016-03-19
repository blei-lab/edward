from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import norm
from scipy import stats

sess = tf.Session()


def _assert_eq(res_ed, res_true):
    with sess.as_default():
        assert np.allclose(res_ed.eval(), res_true)


def test_entropy_empty():
    _assert_eq(norm.entropy(), stats.norm.entropy())


def test_entropy_scalar():
    x = tf.constant(1.0)
    _assert_eq(norm.entropy(x), stats.norm.entropy(1.0))


def test_entropy_1d():
    x = tf.ones([1.0])
    _assert_eq(norm.entropy(x), stats.norm.entropy([1.0]))
