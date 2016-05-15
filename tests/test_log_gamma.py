from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.util import log_gamma
from scipy import special

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-3)

def _test(x):
    xtf = tf.constant(x)
    val_true = special.gammaln(x)
    _assert_eq(log_gamma(xtf), val_true)

def test_scalar():
    _test(0.3)
    _test(0.7)

def test_1d():
    _test([0.5, 0.3, 0.8, 0.1])

def test_2d():
    _test(np.array([[0.5, 0.3, 0.8, 0.1],[0.5, 0.3, 0.8, 0.1]]))
