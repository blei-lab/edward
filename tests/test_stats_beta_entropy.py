from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import beta
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_entropy(a, b):
    val_true = stats.beta.entropy(a, b)
    _assert_eq(beta.entropy(a, b), val_true)
    _assert_eq(beta.entropy(tf.constant(a), tf.constant(b)), val_true)
    _assert_eq(beta.entropy(tf.constant([a]), tf.constant(b)), val_true)
    _assert_eq(beta.entropy(tf.constant(a), tf.constant([b])), val_true)
    _assert_eq(beta.entropy(tf.constant([a]), tf.constant([b])), val_true)

def test_entropy_scalar():
    _test_entropy(a=0.5, b=0.5)
    _test_entropy(a=1.0, b=1.0)
    _test_entropy(a=0.5, b=5.0)
    _test_entropy(a=5.0, b=0.5)

def test_entropy_1d():
    _test_entropy([0.5, 0.3, 0.8, 0.1], [0.5, 0.3, 0.8, 0.1])
