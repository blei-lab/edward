from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import gamma
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_entropy(a, scale=1):
    val_true = stats.gamma.entropy(a, scale=scale)
    _assert_eq(gamma.entropy(a, scale), val_true)
    _assert_eq(gamma.entropy(tf.constant(a), tf.constant(scale)), val_true)
    _assert_eq(gamma.entropy(tf.constant([a]), tf.constant(scale)), val_true)
    _assert_eq(gamma.entropy(tf.constant(a), tf.constant([scale])), val_true)
    _assert_eq(gamma.entropy(tf.constant([a]), tf.constant([scale])), val_true)

def test_entropy_scalar():
    _test_entropy(a=1.0, scale=1.0)
    _test_entropy(a=1.0, scale=1.0)

    _test_entropy(a=0.5, scale=5.0)
    _test_entropy(a=5.0, scale=0.5)

def test_entropy_1d():
    _test_entropy(a=[0.5, 1.2, 5.3, 8.7], scale=[0.5, 1.2, 5.3, 8.7])
